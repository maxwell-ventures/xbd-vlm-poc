#!/usr/bin/env python3
"""Stage 4 — emit chat-format training data from the label file.

Templated end to end. Never hand-edited; regenerate rather than patch.

    python scripts/build_dataset.py --per-class 1500 --val-per-class 200 \
        --test-per-class 400 --out data/processed

## Class balance

xBD is roughly 80% no-damage. A model trained on that distribution learns the
prior, answers "no-damage", and scores well on naive accuracy while being
useless. So the training set is sampled to a per-class cap.

The *test* set is sampled the same way, and that is a choice with a consequence:
the reported accuracy is accuracy on a balanced test set, which is NOT the
accuracy you would see over a real disaster's building stock. It is the right
choice for measuring per-class skill on rare classes with a sample size that
supports the number, and the wrong one for claiming deployment performance. Say
so in the writeup; do not quote overall accuracy as an operational figure.

Selection is deterministic: rows are ordered by a hash of their uid and the
first N per class are taken. Same inputs, same dataset, every time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xbd_vlm.prompts import (  # noqa: E402
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    UNKNOWN_EVENT_RATE,
    build_evidence,
    build_prompt,
    template_fingerprint,
)
from xbd_vlm.sampling import select, stable_fraction  # noqa: E402
from xbd_vlm.schema import (  # noqa: E402
    DAMAGE_GRADES,
    SCHEMA_VERSION,
    Assessment,
    priority_for,
)


def portable(path: str) -> str:
    """Store chip paths relative to the working directory where possible.

    The dataset is built in one place and consumed in another (laptop -> pod).
    Absolute paths baked into the jsonl break on arrival with a confusing
    file-not-found rather than an obvious one.
    """
    try:
        rel = os.path.relpath(path, os.getcwd())
    except ValueError:  # different drive on Windows
        return path
    return path if rel.startswith("..") else rel


def make_example(row: dict, use_pre: bool) -> dict | None:
    grade = row["damage_grade"]
    uid = row["uid"]

    images = [portable(row["chip_post"])]
    has_pre = bool(use_pre and row.get("chip_pre"))
    if has_pre:
        images = [portable(row["chip_pre"]), portable(row["chip_post"])]

    days = int(row["days_since_event"]) if row["days_since_event"] else None

    # Withhold event type on a deterministic slice so the prompt field carries
    # real information rather than being redundant with the pixels.
    masked = stable_fraction(uid, "event-mask") < UNKNOWN_EVENT_RATE
    event_type = "unknown" if masked else row["event_type"]

    answer = Assessment(
        damage=grade,
        evidence=build_evidence(grade, event_type, uid),
        priority=priority_for(grade, days),
    )

    user_content = [{"type": "image"} for _ in images]
    user_content.append(
        {
            "type": "text",
            "text": build_prompt(
                event_type=event_type,
                days_since=days,
                structure_type=None,
                has_pre_image=has_pre,
            ),
        }
    )

    return {
        "id": uid,
        "images": images,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer.to_text()},
        ],
        "meta": {
            "disaster": row["disaster"],
            "event_type_true": row["event_type"],
            "event_type_shown": event_type,
            "event_type_masked": masked,
            "days_since_event": days,
            "damage_grade": grade,
            "priority": answer.priority,
            "has_pre_image": has_pre,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    ap.add_argument("--split", type=Path, default=Path("configs/split.json"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--per-class", type=int, default=1500, help="train cap per class")
    ap.add_argument("--val-per-class", type=int, default=200)
    ap.add_argument("--test-per-class", type=int, default=400)
    ap.add_argument(
        "--pre",
        action="store_true",
        help="include the pre-event chip as a second image (run 2)",
    )
    ap.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()

    assignment = json.loads(args.split.read_text())["assignment"]
    with args.labels.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("chip_post")]

    if args.pre:
        missing = sum(1 for r in rows if not r.get("chip_pre"))
        if missing:
            print(
                f"warning: {missing}/{len(rows)} rows have no pre-event chip; "
                "re-run build_chips.py with --pre",
                file=sys.stderr,
            )

    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        sp = assignment.get(r["disaster"])
        if sp:
            by_split[sp].append(r)

    args.out.mkdir(parents=True, exist_ok=True)
    caps = {
        "train": args.per_class,
        "val": args.val_per_class,
        "test": args.test_per_class,
    }
    tag = f"_{args.tag}" if args.tag else ""
    card = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "template_fingerprint": template_fingerprint(),
        "include_pre_image": args.pre,
        "unknown_event_rate": UNKNOWN_EVENT_RATE,
        "caps_per_class": caps,
        "splits": {},
    }

    for sp in ["train", "val", "test"]:
        chosen = select(by_split[sp], caps[sp], salt=sp)
        path = args.out / f"{sp}{tag}.jsonl"
        counts: Counter = Counter()
        with path.open("w") as f:
            for row in chosen:
                ex = make_example(row, use_pre=args.pre)
                if ex is None:
                    continue
                counts[ex["meta"]["damage_grade"]] += 1
                f.write(json.dumps(ex) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        card["splits"][sp] = {
            "file": str(path),
            "n": sum(counts.values()),
            "sha256": digest,
            "per_class": dict(counts),
            "events": sorted({r["disaster"] for r in chosen}),
        }
        dist = "  ".join(f"{g}={counts[g]}" for g in DAMAGE_GRADES)
        print(f"{sp:<6}{sum(counts.values()):>7}  {dist}")
        print(f"       -> {path}  sha256:{digest[:12]}")

    card_path = args.out / f"dataset_card{tag}.json"
    card_path.write_text(json.dumps(card, indent=2) + "\n")
    print(f"\ntemplate fingerprint {card['template_fingerprint']}")
    print(f"wrote {card_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
