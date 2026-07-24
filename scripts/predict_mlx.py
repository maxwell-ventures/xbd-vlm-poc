#!/usr/bin/env python3
"""Run a split through Qwen2.5-VL locally on Apple silicon via MLX.

Same job as predict.py, same output format, different backend. This exists so
the zero-shot baseline can be measured before renting a GPU — if the prompt or
the schema is wrong, that is much cheaper to discover here.

    .venv/bin/python scripts/predict_mlx.py \
        --data data/processed/test_post.jsonl \
        --out outputs/eval/base_post.jsonl --limit 40

Requires the venv (`python3 -m venv .venv && .venv/bin/pip install mlx-vlm`),
not the system interpreter.

Decoding is greedy (temperature 0). The base-vs-tuned comparison has to be
reproducible, and sampling would make the delta partly noise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"


def build_prompt_text(processor, config, ex: dict) -> str:
    """Reconstruct the chat prompt, dropping the assistant turn.

    The messages come from the dataset file, which build_dataset.py wrote from
    prompts.py — so the text here is identical to what training saw, without
    this script knowing anything about the template.
    """
    from mlx_vlm.prompt_utils import apply_chat_template

    msgs = []
    for m in ex["messages"]:
        if m["role"] == "assistant":
            continue
        content = m["content"]
        if isinstance(content, list):
            text = "\n".join(c["text"] for c in content if c.get("type") == "text")
        else:
            text = content
        msgs.append({"role": m["role"], "content": text})

    return apply_chat_template(
        processor, config, msgs, add_generation_prompt=True, num_images=len(ex["images"])
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    # Left at 1 deliberately: batch_generate in mlx-vlm 0.6.7 raises
    # "index N is out of bounds" on batches of single-image prompts. Sequential
    # runs ~0.5 examples/sec, which is fine for a 1.6k-example baseline.
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip ids already present in --out (a long local run can be "
        "interrupted without losing what it already produced)",
    )
    args = ap.parse_args()

    from mlx_vlm import generate, load
    from mlx_vlm.utils import load_config

    print(f"loading {args.model}", file=sys.stderr)
    model, processor = load(args.model, adapter_path=str(args.adapter) if args.adapter else None)
    config = load_config(args.model)

    examples = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    if args.limit:
        examples = examples[: args.limit]

    done: set[str] = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
        examples = [e for e in examples if e["id"] not in done]
        print(f"resuming: {len(done)} already done, {len(examples)} to go", file=sys.stderr)

    mode = "a" if (args.resume and args.out.exists()) else "w"
    started = time.time()
    n = 0

    with args.out.open(mode) as fout:
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i : i + args.batch_size]
            prompts = [build_prompt_text(processor, config, ex) for ex in batch]

            # batch_generate takes a FLAT list of image paths, one per prompt,
            # so it only applies when every example has exactly one image.
            # The pre/post run has two, and falls back to sequential.
            batchable = len(batch) > 1 and all(len(ex["images"]) == 1 for ex in batch)

            if batchable:
                from mlx_vlm import batch_generate

                res = batch_generate(
                    model,
                    processor,
                    images=[ex["images"][0] for ex in batch],
                    prompts=prompts,
                    max_tokens=args.max_new_tokens,
                    temperature=0.0,
                    verbose=False,
                )
                texts = res.texts
            else:
                texts = []
                for ex, prompt in zip(batch, prompts):
                    res = generate(
                        model,
                        processor,
                        prompt,
                        image=ex["images"],
                        max_tokens=args.max_new_tokens,
                        temperature=0.0,
                        verbose=False,
                    )
                    texts.append(res.text)

            for ex, text in zip(batch, texts):
                fout.write(
                    json.dumps(
                        {
                            "id": ex["id"],
                            "true_damage": ex["meta"]["damage_grade"],
                            "true_priority": ex["meta"]["priority"],
                            "generation": (text or "").strip(),
                            "meta": ex["meta"],
                        }
                    )
                    + "\n"
                )
            fout.flush()
            n += len(batch)
            elapsed = time.time() - started
            eta = (len(examples) - n) / (n / elapsed) if n else 0
            print(
                f"  {n}/{len(examples)}  {n/elapsed:.2f}/s  eta {eta/60:.1f} min",
                file=sys.stderr,
            )

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
