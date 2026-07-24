#!/usr/bin/env python3
"""Stage 3 — crop per-building chips from the full tiles.

Deterministic and regenerable: given the same labels.csv and the same flags,
this produces byte-identical chips. Adds `chip_post` and `chip_pre` columns to
the label file.

    python scripts/build_chips.py --labels data/labels.csv --chips data/chips

## Chipping decisions (documented, and worth ablating)

* **Adaptive window.** The crop side is the building's larger bbox dimension
  times `--context`, clamped to [`--min-side`, `--max-side`], then resized to
  `--size`. This guarantees the target building is fully visible with
  neighbourhood around it.

  The cost: because every chip is resized to the same output size, apparent
  scale varies between chips, so the model cannot read absolute building size
  off the image. `--mode fixed` uses a constant ground window instead, which
  preserves scale but clips large structures. Adaptive is the default because
  cropping the subject is worse than losing the scale cue.

* **Which building?** A chip usually contains several structures. The target is
  centred and the prompt says so. `--outline` additionally draws the polygon,
  which removes the ambiguity entirely but paints a non-photographic cue into
  the pixels — a shortcut the model will happily learn, and one that would not
  exist at inference time on unlabelled imagery. Off by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xbd_vlm.sampling import select_all_splits  # noqa: E402

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    print("pillow is required:  pip install pillow", file=sys.stderr)
    raise


def window_for(row: dict, args) -> tuple[int, int, int, int]:
    """Crop box (left, top, right, bottom), clamped to the tile."""
    cx = float(row["centroid_x"])
    cy = float(row["centroid_y"])
    w = int(float(row["img_width"] or 1024))
    h = int(float(row["img_height"] or 1024))

    if args.mode == "fixed":
        side = args.min_side
    else:
        bw = float(row["bbox_x1"]) - float(row["bbox_x0"])
        bh = float(row["bbox_y1"]) - float(row["bbox_y0"])
        side = max(bw, bh) * args.context
        side = max(args.min_side, min(args.max_side, side))

    half = side / 2.0
    left = cx - half
    top = cy - half
    # Slide the window back inside the tile rather than shrinking it, so the
    # output aspect ratio stays square.
    left = max(0.0, min(left, w - side))
    top = max(0.0, min(top, h - side))
    return (
        int(round(left)),
        int(round(top)),
        int(round(left + side)),
        int(round(top + side)),
    )


def crop_one(
    img: "Image.Image", box, row: dict, args, outline: bool
) -> "Image.Image":
    chip = img.crop(box)
    if outline:
        draw = ImageDraw.Draw(chip)
        draw.rectangle(
            [
                float(row["bbox_x0"]) - box[0],
                float(row["bbox_y0"]) - box[1],
                float(row["bbox_x1"]) - box[0],
                float(row["bbox_y1"]) - box[1],
            ],
            outline=(255, 0, 0),
            width=2,
        )
    return chip.resize((args.size, args.size), Image.BICUBIC)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    ap.add_argument("--out", type=Path, default=None, help="default: in place")
    ap.add_argument("--chips", type=Path, default=Path("data/chips"))
    ap.add_argument("--mode", choices=["adaptive", "fixed"], default="adaptive")
    ap.add_argument("--size", type=int, default=448, help="output px (multiple of 28)")
    ap.add_argument("--context", type=float, default=3.0)
    ap.add_argument("--min-side", type=int, default=128)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--min-area", type=float, default=40.0, help="skip specks")
    ap.add_argument("--pre", action="store_true", help="also chip the pre-event tile")
    ap.add_argument("--outline", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", type=Path, default=Path("configs/split.json"))
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--val-per-class", type=int, default=200)
    ap.add_argument("--test-per-class", type=int, default=400)
    ap.add_argument(
        "--all",
        action="store_true",
        help="chip every building instead of the sampled subset (very large)",
    )
    args = ap.parse_args()

    out_path = args.out or args.labels
    with args.labels.open() as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    # Chip only the buildings the dataset will actually reference. xBD has
    # ~850k polygons; this project uses a few thousand. Chipping everything
    # would produce two orders of magnitude more data than the training set
    # needs, and the sampling is deterministic, so build_dataset.py selects the
    # identical rows without the two scripts having to agree on anything but
    # the caps.
    selected: set[str] | None = None
    if not args.all:
        if not args.split.exists():
            raise SystemExit(
                f"{args.split} not found — run scripts/split.py first, or pass "
                "--all to chip every building (expect hundreds of GB)."
            )
        assignment = json.loads(args.split.read_text())["assignment"]
        caps = {
            "train": args.per_class,
            "val": args.val_per_class,
            "test": args.test_per_class,
        }
        chosen = select_all_splits(rows, assignment, caps)
        selected = {r["uid"] for split_rows in chosen.values() for r in split_rows}
        summary = "  ".join(f"{sp}={len(v)}" for sp, v in sorted(chosen.items()))
        print(f"chipping {len(selected)} of {len(rows)} buildings   {summary}")

    # Sort by tile so each full image is opened exactly once.
    rows.sort(key=lambda r: (r["post_image"], r["uid"]))

    stats = Counter()
    open_path, open_img, open_pre = None, None, None

    for i, row in enumerate(rows, 1):
        row["chip_post"] = ""
        row["chip_pre"] = ""

        if selected is not None and row["uid"] not in selected:
            stats["not_selected"] += 1
            continue

        if float(row["area_px"] or 0) < args.min_area:
            stats["skipped_tiny"] += 1
            continue

        post = Path(row["post_image"])
        if not post.exists():
            stats["missing_tile"] += 1
            continue

        if str(post) != open_path:
            open_img = Image.open(post).convert("RGB")
            open_path = str(post)
            open_pre = None
            if args.pre and row["pre_image"] and Path(row["pre_image"]).exists():
                open_pre = Image.open(row["pre_image"]).convert("RGB")

        box = window_for(row, args)
        dest_dir = args.chips / row["disaster"]
        dest_dir.mkdir(parents=True, exist_ok=True)

        post_chip = dest_dir / f"{row['uid']}_post.png"
        crop_one(open_img, box, row, args, args.outline).save(post_chip)
        row["chip_post"] = str(post_chip)
        stats["chipped"] += 1

        if open_pre is not None:
            pre_chip = dest_dir / f"{row['uid']}_pre.png"
            crop_one(open_pre, box, row, args, args.outline).save(pre_chip)
            row["chip_pre"] = str(pre_chip)
            stats["chipped_pre"] += 1

        if i % 1000 == 0:
            print(f"  {i}/{len(rows)}", file=sys.stderr)

    fields = list(rows[0].keys())
    tmp = out_path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(out_path)

    print(f"\nchips -> {args.chips}")
    print(f"labels -> {out_path}")
    for k, v in stats.most_common():
        print(f"  {k:<16}{v:>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
