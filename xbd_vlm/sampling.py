"""Deterministic, hash-based selection of which buildings to use.

Shared by `build_chips.py` and `build_dataset.py` so both arrive at exactly the
same rows without having to communicate. Selection depends only on the uid and
the split name, so it is stable across machines and across reruns.

That property is what lets chipping happen *before* dataset building while still
chipping only the buildings the dataset will actually reference — the difference
between a few GB of crops and a few hundred.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

from .schema import DAMAGE_GRADES


def stable_fraction(uid: str, salt: str) -> float:
    """Uniform in [0,1), stable across runs, machines and Python versions."""
    h = hashlib.sha1(f"{salt}|{uid}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def select(rows: list[dict], per_class: int, salt: str) -> list[dict]:
    """Take up to `per_class` rows of each damage grade, deterministically.

    Rows are ordered by hash and the first N of each class are taken, so raising
    the cap later is purely additive — the previously selected rows stay
    selected, and their chips stay valid.
    """
    if per_class <= 0:
        return list(rows)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r["damage_grade"]].append(r)
    picked = []
    for grade in DAMAGE_GRADES:
        ordered = sorted(buckets[grade], key=lambda r: stable_fraction(r["uid"], salt))
        picked.extend(ordered[:per_class])
    picked.sort(key=lambda r: stable_fraction(r["uid"], salt + "-order"))
    return picked


def select_all_splits(
    rows: list[dict], assignment: dict[str, str], caps: dict[str, int]
) -> dict[str, list[dict]]:
    """Apply per-split caps to rows grouped by their event's split."""
    by_split: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        sp = assignment.get(r["disaster"])
        if sp:
            by_split[sp].append(r)
    return {sp: select(by_split[sp], caps.get(sp, 0), salt=sp) for sp in by_split}
