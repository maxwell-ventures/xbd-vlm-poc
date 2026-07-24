#!/usr/bin/env python3
"""Run a test split through the model and dump raw generations. GPU side.

    # zero-shot baseline — the number the whole project argues against
    python scripts/predict.py --data data/processed/test.jsonl \
        --out outputs/eval/base.jsonl

    # tuned
    python scripts/predict.py --data data/processed/test.jsonl \
        --adapter outputs/adapters/run1 --out outputs/eval/tuned.jsonl

Writes generations only. Scoring happens in evaluate.py, off the GPU, so a
schema fix never costs another inference pass.

The prompt is rebuilt from the dataset file, which was itself built by
build_dataset.py from prompts.py. Base and tuned therefore see byte-identical
input. If they did not, the delta would be measuring prompt differences.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xbd_vlm.prompts import template_fingerprint  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def check_fingerprint(adapter: Path | None) -> None:
    """Refuse to run an adapter against a template it was not trained on.

    This is the failure mode the brief warns about: inference silently
    reconstructs the prompt differently, performance drops, and there is nothing
    in the logs to explain it.
    """
    if adapter is None:
        return
    cfg_path = adapter / "run_config.json"
    if not cfg_path.exists():
        print(f"warning: no run_config.json in {adapter}", file=sys.stderr)
        return
    trained = json.loads(cfg_path.read_text()).get("template_fingerprint")
    current = template_fingerprint()
    if trained and trained != current:
        raise SystemExit(
            f"template mismatch\n"
            f"  adapter trained on: {trained}\n"
            f"  current prompts.py: {current}\n"
            f"Check out the prompts.py that produced the adapter, or retrain."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    check_fingerprint(args.adapter)

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    # transformers 4.51 names this torch_dtype; the bare `dtype` alias is 5.x-only.
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(args.model)
    # Batched GENERATION requires left-padding: with right-padding the model
    # continues from pad tokens for every sequence shorter than the longest in
    # the batch, corrupting their output. (Training is the opposite — the
    # collator in train.py uses right-padding on purpose.)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, **kwargs)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    examples = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    if args.limit:
        examples = examples[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    with args.out.open("w") as fout:
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i : i + args.batch_size]
            texts, images = [], []
            for ex in batch:
                # Drop the assistant turn — that is the answer we are asking for.
                msgs = [m for m in ex["messages"] if m["role"] != "assistant"]
                texts.append(
                    processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                )
                images.append([Image.open(p).convert("RGB") for p in ex["images"]])

            inputs = processor(
                text=texts, images=images, return_tensors="pt", padding=True
            ).to(model.device)

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,  # greedy: the comparison must be reproducible
                )

            for ex, seq, in_ids in zip(batch, out, inputs["input_ids"]):
                text = processor.decode(
                    seq[len(in_ids) :], skip_special_tokens=True
                ).strip()
                fout.write(
                    json.dumps(
                        {
                            "id": ex["id"],
                            "true_damage": ex["meta"]["damage_grade"],
                            "true_priority": ex["meta"]["priority"],
                            "generation": text,
                            "meta": ex["meta"],
                        }
                    )
                    + "\n"
                )
            done = min(i + args.batch_size, len(examples))
            rate = done / (time.time() - started)
            print(f"  {done}/{len(examples)}  {rate:.1f}/s", file=sys.stderr)

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
