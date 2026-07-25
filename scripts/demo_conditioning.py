#!/usr/bin/env python3
"""The context-conditioning demo. Local inference on Apple silicon.

Holds ONE post-event image fixed and varies only the stated context (event type,
then days-since), for both the base and the tuned 3B model. Shows that the tuned
model's assessment tracks the text while the base model's does not.

Honesty note carried into the writeup: the conditioning the tuned model displays
is one it was TRAINED to display — evidence phrasing is templated on event type,
priority is a rule on days-since. So this demonstrates learned conditioning, not
emergent reasoning. It is still the thing a classifier cannot do: change its
output because a word in the prompt changed.

    .venv/bin/python scripts/demo_conditioning.py --uid <uid> --disaster hurricane-michael
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xbd_vlm.prompts import SYSTEM_PROMPT, build_prompt

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# The variants. Fixed image throughout.
EVENT_VARIANTS = ["hurricane", "flood", "wildfire", "earthquake"]  # days fixed
FIXED_DAYS = 5
DAYS_VARIANTS = [3, 45, 200]                                       # event fixed
FIXED_EVENT = "hurricane"


def variants():
    out = []
    for ev in EVENT_VARIANTS:
        out.append({"kind": "event", "event_type": ev, "days": FIXED_DAYS,
                    "label": f"event = {ev}"})
    for dz in DAYS_VARIANTS:
        out.append({"kind": "days", "event_type": FIXED_EVENT, "days": dz,
                    "label": f"days = {dz}"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uid", required=True)
    ap.add_argument("--disaster", required=True)
    ap.add_argument("--adapter", type=Path, default=Path("outputs/adapters/run1_post"))
    ap.add_argument("--out", type=Path, default=Path("outputs/eval/demo_conditioning.json"))
    ap.add_argument("--max-new-tokens", type=int, default=90)
    args = ap.parse_args()

    chip = Path(f"data/chips/{args.disaster}/{args.uid}_post.png")
    if not chip.exists():
        raise SystemExit(f"chip not found: {chip}")

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"device={device} dtype={dtype}", file=sys.stderr)

    processor = AutoProcessor.from_pretrained(MODEL)
    processor.tokenizer.padding_side = "left"
    img = Image.open(chip).convert("RGB")
    vs = variants()

    def run(model, tag):
        results = []
        for v in vs:
            prompt = build_prompt(v["event_type"], v["days"], None, has_pre_image=False)
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[[img]], return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen = processor.decode(out[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()
            results.append({**v, "generation": gen})
            print(f"  [{tag}] {v['label']:<18} -> {gen.splitlines()[0] if gen else '(empty)'}",
                  file=sys.stderr)
        return results

    print("loading base…", file=sys.stderr)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=dtype).to(device).eval()
    t = time.time(); base_res = run(base, "base"); print(f"base done {time.time()-t:.0f}s", file=sys.stderr)
    del base
    if device == "mps": torch.mps.empty_cache()

    print("loading tuned…", file=sys.stderr)
    from peft import PeftModel
    tm = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=dtype)
    tm = PeftModel.from_pretrained(tm, str(args.adapter)).to(device).eval()
    t = time.time(); tuned_res = run(tm, "tuned"); print(f"tuned done {time.time()-t:.0f}s", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "uid": args.uid, "disaster": args.disaster, "chip": str(chip),
        "fixed_days": FIXED_DAYS, "fixed_event": FIXED_EVENT,
        "base": base_res, "tuned": tuned_res,
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
