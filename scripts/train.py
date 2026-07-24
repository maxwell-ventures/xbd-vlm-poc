#!/usr/bin/env python3
"""Stage 7 — LoRA fine-tune Qwen2.5-VL on the damage-assessment task. GPU side.

    python scripts/train.py \
        --train data/processed/train_post.jsonl \
        --val   data/processed/val_post.jsonl \
        --out   outputs/adapters/run1_post

Produces a PEFT adapter that scripts/predict.py loads directly, plus a
run_config.json capturing everything needed to reproduce and to use it safely.

## What is adapted, and what is frozen

The vision encoder is **frozen**. The bet (see notes/decisions.md) is that it
already represents rubble and burn scars as texture, and the adaptation needed
is in the language model — the mapping from visual features to a damage grade
and to this exact output format.

LoRA adapters are attached to the attention and MLP projections of the language
model only. PEFT freezes everything it does not target, so naming just those
modules freezes the encoder for free.

`--train-projector` additionally trains the multimodal merger (the layer that
bridges vision embeddings into the token space). It is the obvious hedge if the
frozen-encoder bet underperforms, and is off by default to keep run 1 simple.

## Why these hyperparameters

rank 16, alpha 32 (2×rank), lr 1e-4, 2 epochs, cosine schedule with warmup. LoRA
tolerates a learning rate an order of magnitude above full fine-tuning because it
only moves a low-rank correction. Watch eval loss and stop when it turns; on this
balanced ~6k-example set two epochs is usually near the elbow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xbd_vlm.prompts import PROMPT_VERSION, template_fingerprint  # noqa: E402
from xbd_vlm.schema import SCHEMA_VERSION  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# Attention + MLP projections of the language model. These names do not occur in
# the vision tower (whose modules live under `visual.`), so targeting them by
# name leaves the encoder untouched.
LM_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_examples(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_collator(processor):
    """Collate chat examples into a padded batch with the prompt masked out.

    Only the assistant turn contributes to the loss. Everything before it — the
    system prompt, the image tokens, the context, the schema instruction — is set
    to the ignore index, so the model is trained to *produce* the assessment, not
    to reconstruct the question.

    The prompt/answer boundary is found per example by processing the prompt-only
    turns and taking that length. Doing it per example keeps it correct across
    the variable number of image tokens (one image in run 1, two in run 2).
    """
    from PIL import Image

    IGNORE = -100

    def split_messages(ex):
        prompt_msgs, full_msgs = [], []
        for m in ex["messages"]:
            full_msgs.append(m)
            if m["role"] != "assistant":
                prompt_msgs.append(m)
        return prompt_msgs, full_msgs

    def collate(batch):
        import torch

        texts_full, texts_prompt, images_per = [], [], []
        for ex in batch:
            prompt_msgs, full_msgs = split_messages(ex)
            texts_full.append(
                processor.apply_chat_template(
                    full_msgs, tokenize=False, add_generation_prompt=False
                )
            )
            texts_prompt.append(
                processor.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True
                )
            )
            images_per.append([Image.open(p).convert("RGB") for p in ex["images"]])

        model_inputs = processor(
            text=texts_full, images=images_per, return_tensors="pt", padding=True
        )

        labels = model_inputs["input_ids"].clone()
        # Ignore pad positions.
        labels[model_inputs["attention_mask"] == 0] = IGNORE

        # Ignore the prompt prefix of each row. The prompt-only encoding is
        # processed with the same images, so image-token expansion matches and
        # its length is the true boundary.
        for i, ex in enumerate(batch):
            prompt_ids = processor(
                text=[texts_prompt[i]],
                images=[images_per[i]],
                return_tensors="pt",
                padding=False,
            )["input_ids"]
            boundary = prompt_ids.shape[1]
            # Left-padding would shift the boundary; Qwen's processor pads right,
            # so the first `boundary` real tokens are the prompt.
            labels[i, :boundary] = IGNORE

        model_inputs["labels"] = labels
        return model_inputs

    return collate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None, help="pin the base model revision")

    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--train-projector", action="store_true")

    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)

    ap.add_argument("--load-4bit", action="store_true", help="QLoRA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-steps", type=int, default=50)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    # --- provenance, written before training so a crash still leaves a record ---
    args.out.mkdir(parents=True, exist_ok=True)
    train_sha = sha256_file(args.train)
    val_sha = sha256_file(args.val)

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    # The label-boundary masking in the collator assumes real tokens sit at the
    # START of each row (prompt prefix, then answer, then pad). Qwen's tokenizer
    # often defaults to left-padding for generation, which would put pad tokens
    # first and shift the boundary — force right-padding for training.
    processor.tokenizer.padding_side = "right"

    load_kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, **load_kwargs
    )
    model.config.use_cache = False

    # Train the multimodal merger as full weights (small) when asked. modules_to_save
    # keeps the layer trainable and serialises it into the adapter, so predict.py
    # picks it up with no special handling.
    modules_to_save = ["merger"] if args.train_projector else None

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LM_TARGET_MODULES,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = load_examples(args.train)
    val_ds = load_examples(args.val)
    collate = build_collator(processor)

    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,  # the collator produces pixel_values etc.
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
    )

    # run_config.json: written now (pre-train) and refreshed with final loss after.
    # This is the file predict.py checks the template fingerprint against — an
    # adapter without it, or with a stale fingerprint, is a silent-failure risk.
    run_config = {
        "base_model": args.model,
        "base_model_revision": args.revision,
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "template_fingerprint": template_fingerprint(),
        "train_file": str(args.train),
        "train_sha256": train_sha,
        "val_file": str(args.val),
        "val_sha256": val_sha,
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": LM_TARGET_MODULES,
            "train_projector": args.train_projector,
        },
        "optim": {
            "epochs": args.epochs,
            "lr": args.lr,
            "warmup_ratio": args.warmup_ratio,
            "scheduler": "cosine",
            "effective_batch": args.batch_size * args.grad_accum,
            "load_4bit": args.load_4bit,
            "seed": args.seed,
        },
    }
    (args.out / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    result = trainer.train()

    # Save the adapter (best checkpoint, thanks to load_best_model_at_end).
    model.save_pretrained(str(args.out))
    processor.save_pretrained(str(args.out))

    run_config["result"] = {
        "train_loss": result.metrics.get("train_loss"),
        "best_eval_loss": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
    }
    (args.out / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    print(f"\nadapter -> {args.out}")
    print(f"best eval loss: {trainer.state.best_metric}")
    print("next: scripts/predict.py --adapter", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
