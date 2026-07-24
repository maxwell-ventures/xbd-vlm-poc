#!/usr/bin/env python3
"""Stage 5 — assign whole disaster events to train / val / test.

    python scripts/split.py --labels data/labels.csv --out configs/split.json

## Why grouped by event

The obvious split — shuffle buildings, take 80/10/10 — leaks catastrophically
here. Buildings from one tile share the same lighting, sensor, resolution,
architectural vernacular and annotator. A random split puts near-duplicate
neighbours on both sides of the wall, and the reported accuracy measures
memorisation of tiles rather than transfer to a new disaster. Grouping by event
is the only split that answers the question we actually care about: does this
generalise to the next disaster?

This deviates from the official xView2 split, which is drawn within events. The
deviation is deliberate and belongs in the writeup.

## The coverage problem it creates

Nineteen events across seven event types, unevenly. Hold out an entire event and
you may hold out the only example of its type — so the test set asks the model
about a context it has never seen, and the context-conditioning result becomes
untestable rather than merely hard. This script reports coverage per type and
refuses to hide a type that appears in test but not in train.

The output is checked into git. It is a decision, not an artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xbd_vlm.schema import DAMAGE_GRADES  # noqa: E402


def assign(by_type: dict[str, list[tuple[str, int]]]) -> dict[str, str]:
    """Greedy, deterministic: biggest event of each type trains, next tests.

    Events arrive sorted by building count, descending. Assigning the largest to
    train maximises training signal; assigning the second-largest to test keeps
    the test set big enough for per-class numbers to mean anything.
    """
    split_of: dict[str, str] = {}
    for etype in sorted(by_type):
        events = by_type[etype]
        if len(events) == 1:
            split_of[events[0][0]] = "train"
        elif len(events) == 2:
            split_of[events[0][0]] = "train"
            split_of[events[1][0]] = "test"
        else:
            split_of[events[0][0]] = "train"
            split_of[events[1][0]] = "test"
            split_of[events[2][0]] = "val"
            for name, _ in events[3:]:
                split_of[name] = "train"
    return split_of


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    ap.add_argument("--out", type=Path, default=Path("configs/split.json"))
    ap.add_argument(
        "--manual",
        type=Path,
        default=None,
        help="json {event: split} to override the greedy assignment",
    )
    args = ap.parse_args()

    with args.labels.open() as f:
        rows = list(csv.DictReader(f))

    counts: Counter = Counter()
    etype_of: dict[str, str] = {}
    grades: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        counts[r["disaster"]] += 1
        etype_of[r["disaster"]] = r["event_type"]
        grades[r["disaster"]][r["damage_grade"]] += 1

    by_type: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for disaster, n in counts.items():
        by_type[etype_of[disaster]].append((disaster, n))
    for etype in by_type:
        by_type[etype].sort(key=lambda kv: (-kv[1], kv[0]))

    split_of = assign(by_type)
    if args.manual:
        split_of.update(json.loads(args.manual.read_text()))

    # --- report -----------------------------------------------------------
    print(f"{'event':<24}{'type':<20}{'split':<8}{'buildings':>10}")
    for etype in sorted(by_type):
        for disaster, n in by_type[etype]:
            print(f"{disaster:<24}{etype:<20}{split_of[disaster]:<8}{n:>10}")

    print(f"\n{'event type':<20}{'train':>8}{'val':>8}{'test':>8}")
    problems = []
    for etype in sorted(by_type):
        per = Counter(split_of[d] for d, _ in by_type[etype])
        print(f"{etype:<20}{per['train']:>8}{per['val']:>8}{per['test']:>8}")
        if per["test"] and not per["train"]:
            problems.append(f"{etype}: in test but never seen in training")
        if not per["test"]:
            problems.append(f"{etype}: no held-out event — cannot be tested cross-event")

    if not any(sp == "val" for sp in split_of.values()):
        problems.append(
            "no validation events at all — early stopping would have to read the "
            "test set, which converts test into val and inflates the headline"
        )

    print(f"\n{'split':<10}{'buildings':>12}   " + "".join(f"{g[:9]:>12}" for g in DAMAGE_GRADES))
    split_grades: dict[str, Counter] = defaultdict(Counter)
    for disaster, n in counts.items():
        split_grades[split_of[disaster]].update(grades[disaster])
    for sp in ["train", "val", "test"]:
        c = split_grades[sp]
        total = sum(c.values())
        cells = "".join(f"{c[g]:>12}" for g in DAMAGE_GRADES)
        print(f"{sp:<10}{total:>12}   {cells}")
        if total:
            shares = "".join(f"{c[g]/total:>11.1%}" for g in DAMAGE_GRADES)
            print(f"{'':<10}{'':>12}   {shares}")

    if problems:
        print("\nCOVERAGE WARNINGS")
        for p in problems:
            print(f"  ! {p}")
        print(
            "\n  Fix with --manual before training. A type in test but not in\n"
            "  train makes the headline number a zero-shot-transfer result, not\n"
            "  a fine-tuning result, and they are different claims."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "grouping": "event",
                "note": "deviates from the official xView2 split; see docstring",
                "assignment": dict(sorted(split_of.items())),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
