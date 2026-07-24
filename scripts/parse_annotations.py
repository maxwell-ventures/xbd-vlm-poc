#!/usr/bin/env python3
"""Stage 2 — parse xBD GeoJSON annotations into a flat label file.

Walks every `*_post_disaster.json` under the raw data root and emits one CSV row
per graded building polygon. This file is the source of truth for everything
downstream; nothing else reads the raw annotations.

    python scripts/parse_annotations.py --raw data/raw --out data/labels.csv

Expected raw layout (as xBD ships it):

    data/raw/<source>/images/<disaster>_<tile>_post_disaster.png
    data/raw/<source>/labels/<disaster>_<tile>_post_disaster.json

where <source> is train / tier3 / test / hold.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xbd_vlm.events import days_since_event, event_type_for  # noqa: E402
from xbd_vlm.schema import DAMAGE_GRADES, UNGRADED_SUBTYPE  # noqa: E402

FIELDS = [
    "uid",
    "source",
    "disaster",
    "event_type",
    "tile_id",
    "capture_date",
    "days_since_event",
    "damage_grade",
    "post_image",
    "pre_image",
    "img_width",
    "img_height",
    "gsd",
    "centroid_x",
    "centroid_y",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "area_px",
]

# POLYGON ((x y, x y, ...)) — buildings are simple rings, so the first ring is
# the whole story. Kept dependency-free rather than pulling in shapely.
_RING_RE = re.compile(r"\(\(([^)]*)\)\)")


def parse_wkt_ring(wkt: str) -> list[tuple[float, float]] | None:
    m = _RING_RE.search(wkt)
    if not m:
        return None
    pts = []
    for pair in m.group(1).split(","):
        parts = pair.split()
        if len(parts) < 2:
            return None
        pts.append((float(parts[0]), float(parts[1])))
    return pts or None


def ring_stats(pts: list[tuple[float, float]]) -> dict:
    """Centroid, bounding box and area (shoelace) in pixel coordinates."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    area2 = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        area2 += x0 * y1 - x1 * y0
    return {
        "centroid_x": sum(xs) / len(xs),
        "centroid_y": sum(ys) / len(ys),
        "bbox_x0": min(xs),
        "bbox_y0": min(ys),
        "bbox_x1": max(xs),
        "bbox_y1": max(ys),
        "area_px": abs(area2) / 2.0,
    }


def parse_capture_date(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def parse_tile(label_path: Path, source: str) -> tuple[list[dict], Counter]:
    counts = Counter()
    with label_path.open() as f:
        doc = json.load(f)

    meta = doc.get("metadata", {})
    disaster = meta.get("disaster") or label_path.stem.split("_")[0]
    stem = label_path.stem  # <disaster>_<tile>_post_disaster
    tile_id = stem.replace("_post_disaster", "")

    images_dir = label_path.parent.parent / "images"
    post_image = images_dir / f"{stem}.png"
    pre_image = images_dir / f"{stem.replace('_post_', '_pre_')}.png"

    capture = parse_capture_date(meta.get("capture_date"))
    event_type = event_type_for(disaster)
    if event_type == "unknown":
        counts["unknown_disaster"] += 1
    days = days_since_event(disaster, capture)

    rows = []
    for feat in doc.get("features", {}).get("xy", []):
        props = feat.get("properties", {})
        if props.get("feature_type") != "building":
            continue
        subtype = props.get("subtype")
        if subtype in (None, UNGRADED_SUBTYPE):
            counts["ungraded"] += 1
            continue
        if subtype not in DAMAGE_GRADES:
            counts[f"unexpected_subtype:{subtype}"] += 1
            continue

        pts = parse_wkt_ring(feat.get("wkt", ""))
        if pts is None:
            counts["bad_geometry"] += 1
            continue

        stats = ring_stats(pts)
        counts[subtype] += 1
        rows.append(
            {
                "uid": props.get("uid") or f"{tile_id}:{len(rows)}",
                "source": source,
                "disaster": disaster,
                "event_type": event_type,
                "tile_id": tile_id,
                "capture_date": capture.isoformat() if capture else "",
                "days_since_event": "" if days is None else days,
                "damage_grade": subtype,
                "post_image": str(post_image),
                "pre_image": str(pre_image) if pre_image.exists() else "",
                "img_width": meta.get("width", ""),
                "img_height": meta.get("height", ""),
                "gsd": meta.get("gsd", ""),
                **{k: round(v, 2) for k, v in stats.items()},
            }
        )
    return rows, counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/labels.csv"))
    ap.add_argument(
        "--limit-tiles",
        type=int,
        default=0,
        help="parse at most N tiles (for a quick smoke run)",
    )
    args = ap.parse_args()

    label_files = sorted(args.raw.glob("*/labels/*_post_disaster.json"))
    if not label_files:
        print(f"no post-disaster label files under {args.raw}", file=sys.stderr)
        print(
            "expected data/raw/<train|tier3|test|hold>/labels/*.json",
            file=sys.stderr,
        )
        return 1
    if args.limit_tiles:
        label_files = label_files[: args.limit_tiles]

    totals = Counter()
    by_disaster = Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, path in enumerate(label_files, 1):
            source = path.parent.parent.name
            rows, counts = parse_tile(path, source)
            totals.update(counts)
            for row in rows:
                by_disaster[row["disaster"]] += 1
                writer.writerow(row)
            if i % 200 == 0:
                print(f"  {i}/{len(label_files)} tiles", file=sys.stderr)

    n = sum(totals[g] for g in DAMAGE_GRADES)
    print(f"\nwrote {n} graded buildings from {len(label_files)} tiles -> {args.out}")
    print("\ndamage grade distribution")
    for g in DAMAGE_GRADES:
        share = totals[g] / n if n else 0
        print(f"  {g:<16}{totals[g]:>9}  {share:6.1%}")
    skipped = {k: v for k, v in totals.items() if k not in DAMAGE_GRADES}
    if skipped:
        print("\nskipped")
        for k, v in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<24}{v:>9}")
    print("\nbuildings per event")
    for d, c in sorted(by_disaster.items(), key=lambda kv: -kv[1]):
        print(f"  {d:<24}{c:>9}  ({event_type_for(d)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
