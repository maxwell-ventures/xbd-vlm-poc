#!/usr/bin/env python3
"""Stage 7 — score model generations. No GPU, no model, no network.

Reads a predictions file (raw generations), parses them through the one schema
parser, and emits the metrics table plus per-example rows for inspection.

    python scripts/evaluate.py --pred outputs/eval/base.jsonl
    python scripts/evaluate.py --pred outputs/eval/tuned.jsonl \
                               --baseline outputs/eval/base.jsonl

Predictions file — one JSON object per line:

    {"id": "...", "true_damage": "major-damage", "true_priority": "high",
     "generation": "DAMAGE: ...\\nEVIDENCE: ...\\nPRIORITY: ...",
     "meta": {...}}

Keeping scoring separate from inference is deliberate: the metric code can be
developed and tested before a single GPU hour is spent, and re-scoring after a
schema fix costs nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xbd_vlm.metrics import format_delta, format_report, score  # noqa: E402
from xbd_vlm.schema import parse_response  # noqa: E402


def load(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            parsed = parse_response(raw.get("generation", ""))
            records.append(
                {
                    "id": raw.get("id"),
                    "true_damage": raw["true_damage"],
                    "true_priority": raw.get("true_priority"),
                    "pred_damage": parsed.assessment.damage if parsed.ok else None,
                    "pred_priority": parsed.assessment.priority if parsed.ok else None,
                    "pred_evidence": parsed.assessment.evidence if parsed.ok else None,
                    "parse_reason": parsed.reason,
                    "generation": raw.get("generation", ""),
                    "meta": raw.get("meta", {}),
                }
            )
    return records


def slice_report(records: list[dict], key: str, title: str) -> str:
    """Break the metrics down by a meta field — event type, masked-context, etc.

    Where the model fails matters more than the aggregate. Per-event-type
    numbers are how underrepresented-event degradation shows up.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        v = r["meta"].get(key)
        if v is not None:
            groups[str(v)].append(r)
    if len(groups) < 2:
        return ""
    lines = [f"--- by {title} ---", f"  {'group':<22}{'n':>6}{'acc':>8}{'macroF1':>9}{'QWK':>8}{'unpar':>8}"]
    for g in sorted(groups):
        m = score(groups[g])
        lines.append(
            f"  {g:<22}{m['n_total']:>6}{m['accuracy_all']:>8.3f}"
            f"{m['macro_f1']:>9.3f}{m['qwk']:>8.3f}{m['unparseable_rate']:>8.1%}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", type=Path, default=None, help="write metrics json here")
    args = ap.parse_args()

    records = load(args.pred)
    if not records:
        print(f"no records in {args.pred}", file=sys.stderr)
        return 1

    name = args.name or args.pred.stem
    metrics = score(records)
    print(format_report(name, metrics))

    for key, title in [
        ("event_type_true", "event type"),
        ("event_type_masked", "context withheld"),
        ("has_pre_image", "pre-event image"),
    ]:
        block = slice_report(records, key, title)
        if block:
            print()
            print(block)

    if args.baseline:
        base_records = load(args.baseline)
        base_metrics = score(base_records)
        print()
        print(format_report(args.baseline.stem, base_metrics))
        print()
        print(format_delta(base_metrics, metrics))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"name": name, **metrics}, indent=2) + "\n")

        # Per-example rows, sorted worst-first, for eyeballing failures.
        rows_path = args.out.with_name(args.out.stem + "_examples.jsonl")
        with rows_path.open("w") as f:
            for r in sorted(
                records,
                key=lambda r: -(0 if r["pred_damage"] else 9),
            ):
                f.write(json.dumps(r) + "\n")
        print(f"\nwrote {args.out}\nwrote {rows_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
