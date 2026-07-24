#!/usr/bin/env python3
"""Exercise the whole no-GPU path on synthetic data.

Run this before touching the dataset. It proves the schema round-trips, the
parser catches the failure modes a real zero-shot model will produce, and the
metrics respond the way they should to known-bad models.

    python scripts/smoke_test.py

The third case is the important one: a model that always answers "no-damage"
scores 80% accuracy on a naturally distributed test set and 0.0 QWK. If that
does not hold, the metrics are wrong, and every number downstream is wrong.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xbd_vlm.metrics import format_delta, format_report, score
from xbd_vlm.prompts import build_evidence, build_prompt, template_fingerprint
from xbd_vlm.schema import (
    DAMAGE_GRADES,
    Assessment,
    parse_response,
    priority_for,
    schema_description,
)

FAILS = []


def check(label: str, cond: bool) -> None:
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}")
    if not cond:
        FAILS.append(label)


def main() -> int:
    print("\n1. schema round-trip")
    a = Assessment("major-damage", "Roof gone; debris scattered.", "high")
    r = parse_response(a.to_text())
    check("clean output parses", r.ok)
    check("damage preserved", r.ok and r.assessment.damage == "major-damage")
    check("priority preserved", r.ok and r.assessment.priority == "high")

    print("\n2. parser handles what a zero-shot model actually emits")
    cases = [
        ("chatty preamble", "Sure! Here is my assessment:\n\nDAMAGE: destroyed\nEVIDENCE: Total collapse.\nPRIORITY: critical", True),
        ("markdown bold", "**DAMAGE:** minor-damage\n**EVIDENCE:** Some roof loss.\n**PRIORITY:** moderate", True),
        ("alias vocabulary", "DAMAGE: Major Damage\nEVIDENCE: Roof failure.\nPRIORITY: Urgent", True),
        ("lowercase keys", "damage: no-damage\nevidence: Intact.\npriority: none", True),
        ("free prose", "The building appears to be severely damaged.", False),
        ("missing field", "DAMAGE: destroyed\nPRIORITY: critical", False),
        ("invented grade", "DAMAGE: catastrophic\nEVIDENCE: x\nPRIORITY: high", False),
        ("empty", "", False),
    ]
    for label, text, want in cases:
        got = parse_response(text)
        check(f"{label} -> {'parses' if want else 'rejected'}", got.ok == want)

    print("\n3. metrics respond correctly to known-bad models")
    rng = random.Random(0)
    # A naturally skewed test set, roughly xBD's real distribution.
    truth = rng.choices(DAMAGE_GRADES, weights=[80, 8, 7, 5], k=1000)

    def records(predict):
        out = []
        for i, t in enumerate(truth):
            p = predict(i, t)
            gen = (
                Assessment(p, "x.", priority_for(p, 3)).to_text()
                if p
                else "I cannot assess this image."
            )
            out.append(
                {
                    "id": str(i),
                    "true_damage": t,
                    "true_priority": priority_for(t, 3),
                    "generation": gen,
                }
            )
        return out

    def scored(recs):
        parsed = []
        for r in recs:
            pr = parse_response(r["generation"])
            parsed.append(
                {
                    "true_damage": r["true_damage"],
                    "true_priority": r["true_priority"],
                    "pred_damage": pr.assessment.damage if pr.ok else None,
                    "pred_priority": pr.assessment.priority if pr.ok else None,
                    "parse_reason": pr.reason,
                    "meta": {},
                }
            )
        return score(parsed)

    always = scored(records(lambda i, t: "no-damage"))
    check(f"majority-class accuracy is high ({always['accuracy_all']:.2f})", always["accuracy_all"] > 0.7)
    check(f"...but macro F1 is low ({always['macro_f1']:.2f})", always["macro_f1"] < 0.3)
    check(f"...and QWK is ~0 ({always['qwk']:.2f})", abs(always["qwk"]) < 0.05)

    perfect = scored(records(lambda i, t: t))
    check("perfect model: accuracy 1.0", perfect["accuracy_all"] == 1.0)
    check("perfect model: QWK 1.0", abs(perfect["qwk"] - 1.0) < 1e-9)

    broken = scored(records(lambda i, t: None if i % 4 == 0 else t))
    check(f"unparseable rate tracked ({broken['unparseable_rate']:.2f})", abs(broken["unparseable_rate"] - 0.25) < 0.02)
    check("accuracy_all charged for failures", broken["accuracy_all"] < broken["accuracy_parsed"])

    adjacent = scored(
        records(lambda i, t: DAMAGE_GRADES[min(3, DAMAGE_GRADES.index(t) + 1)])
    )
    distant = scored(records(lambda i, t: DAMAGE_GRADES[3 - DAMAGE_GRADES.index(t)]))
    check("adjacent confusion beats distant on MAE", adjacent["mae_ordinal"] < distant["mae_ordinal"])
    check("adjacent confusion beats distant on QWK", adjacent["qwk"] > distant["qwk"])

    print("\n4. reports render")
    print()
    print(format_report("always-no-damage", always))
    print()
    print(format_delta(always, adjacent))

    print("\n5. prompt + target sample")
    print()
    print("--- system+user ---")
    print(build_prompt("wildfire", 4, "residential", has_pre_image=False))
    print()
    print("--- target ---")
    print(
        Assessment(
            "major-damage", build_evidence("major-damage", "wildfire", "demo-uid"), priority_for("major-damage", 4)
        ).to_text()
    )
    print()
    print("--- same building, event type swapped to flood ---")
    print(
        Assessment(
            "major-damage", build_evidence("major-damage", "flood", "demo-uid"), priority_for("major-damage", 4)
        ).to_text()
    )
    print()
    print("--- same building, 200 days later ---")
    print(
        Assessment(
            "major-damage", build_evidence("major-damage", "wildfire", "demo-uid"), priority_for("major-damage", 200)
        ).to_text()
    )
    print(f"\ntemplate fingerprint: {template_fingerprint()}")
    assert schema_description()

    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
