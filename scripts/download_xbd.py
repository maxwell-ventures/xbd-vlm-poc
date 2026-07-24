#!/usr/bin/env python3
"""Stage 1 — fetch xBD, verify it, and record what was fetched.

    # 1. log in at https://xview2.org, accept the terms, open the download page
    # 2. copy the download links (right-click -> copy link address)
    # 3. paste them, one per line, into configs/xbd_urls.txt
    # 4. run:
    python scripts/download_xbd.py --dest data/raw

## Why links rather than a username and password

The links xView2 hands out are signed and time-limited. Copying them avoids
storing a password anywhere and avoids a login scraper that breaks the next time
the site's markup changes. The cost is that links expire — if a download 403s,
refresh the page and re-copy. `configs/xbd_urls.txt` is gitignored precisely
because a signed link is a credential.

## What it does

* refuses to start if the destination cannot hold the archives plus their
  extracted contents (roughly 2.2x the archive size)
* resumes interrupted downloads rather than restarting them
* records sha256 and byte size of every archive into notes/dataset.md, so the
  exact data version is reproducible
* verifies the extracted layout is what the rest of the pipeline expects
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Approximate archive sizes, for the disk precheck only. Verify against the
# site; these are used to warn early, not to validate.
KNOWN_SIZES_GB = {
    "train": 8.0,
    "tier3": 17.0,
    "test": 1.7,
    "hold": 1.7,
}


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1024**3


def guess_archive_gb(urls: list[str]) -> float:
    total = 0.0
    for u in urls:
        name = u.split("?")[0].rsplit("/", 1)[-1].lower()
        total += next(
            (gb for key, gb in KNOWN_SIZES_GB.items() if key in name), 2.0
        )
    return total


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def url_filename(url: str) -> str:
    return url.split("?")[0].rsplit("/", 1)[-1] or "xbd_archive.tar.gz"


_PART_RE = re.compile(r"^(?P<base>.+?)\.part-[a-z]{2,}$", re.I)


def plan(urls: list[str]) -> list[tuple[str, list[str]]]:
    """Group URLs into download jobs, joining multipart archives.

    xView2 serves the full GeoTIFF release as `name.tgz.part-aa`, `.part-ab`,
    ... Those are raw byte slices, not independent archives: they must be
    concatenated in order before anything can read them.
    """
    jobs: dict[str, list[str]] = {}
    for url in urls:
        name = url_filename(url)
        m = _PART_RE.match(name)
        key = m.group("base") if m else name
        jobs.setdefault(key, []).append(url)
    for key in jobs:
        jobs[key].sort(key=url_filename)
    return list(jobs.items())


def preflight(jobs: list[tuple[str, list[str]]], allow_geotiff: bool) -> None:
    """Catch the wrong-dataset mistake before gigabytes move.

    The GeoTIFF release and the challenge release are both prominent on the
    download page, and only the challenge one matches this pipeline: PNG tiles
    in <source>/images with GeoJSON in <source>/labels. GeoTIFF needs
    rasterio/GDAL to read, which build_chips.py does not use.
    """
    geotiff = [name for name, _ in jobs if "geotiff" in name.lower()]
    if geotiff and not allow_geotiff:
        raise SystemExit(
            "\nThese look like the full GeoTIFF release:\n"
            + "".join(f"  {n}\n" for n in geotiff)
            + "\nThis pipeline expects the CHALLENGE (PNG) downloads instead —\n"
            "  train_images_labels_targets.tar.gz  (and test / hold / tier3)\n"
            "which extract to <source>/images/*.png + <source>/labels/*.json.\n\n"
            "GeoTIFF would need rasterio/GDAL in build_chips.py, is much larger,\n"
            "and has a different directory layout.\n\n"
            "Re-copy the challenge links into configs/xbd_urls.txt, or pass\n"
            "--allow-geotiff if you really mean it."
        )


def join_parts(parts: list[Path], target: Path) -> Path:
    """Concatenate multipart downloads in lexical order."""
    print(f"    joining {len(parts)} parts -> {target.name}")
    with target.open("wb") as out:
        for part in parts:
            with part.open("rb") as f:
                shutil.copyfileobj(f, out, length=1 << 22)
    for part in parts:
        part.unlink()
    return target


def download(url: str, dest_dir: Path) -> Path:
    name = url.split("?")[0].rsplit("/", 1)[-1] or "xbd_archive.tar.gz"
    out = dest_dir / name
    print(f"\n--> {name}")
    # -C - resumes a partial file; --fail turns an expired link into a non-zero
    # exit rather than a 12-byte HTML error page saved as a tarball.
    result = subprocess.run(
        ["curl", "-L", "--fail", "-C", "-", "-o", str(out), url],
        check=False,
    )
    if result.returncode == 33:
        # Server refused a ranged request; start over.
        out.unlink(missing_ok=True)
        result = subprocess.run(
            ["curl", "-L", "--fail", "-o", str(out), url], check=False
        )
    if result.returncode != 0:
        raise SystemExit(
            f"\ncurl failed ({result.returncode}) on {name}.\n"
            "If this is a 403, the signed link has expired — refresh the "
            "xView2 download page and re-copy it into configs/xbd_urls.txt."
        )
    return out


def extract(archive: Path, dest: Path) -> None:
    print(f"    extracting {archive.name}")
    subprocess.run(["tar", "-xf", str(archive), "-C", str(dest)], check=True)


def verify_layout(dest: Path) -> list[str]:
    """Confirm the pipeline's expected layout actually exists."""
    problems = []
    sources = [p for p in dest.iterdir() if p.is_dir()]
    if not sources:
        return ["nothing extracted under " + str(dest)]
    found_any = False
    for src in sorted(sources):
        labels = src / "labels"
        images = src / "images"
        if not labels.is_dir() or not images.is_dir():
            continue
        found_any = True
        n_post = len(list(labels.glob("*_post_disaster.json")))
        n_img = len(list(images.glob("*_post_disaster.png")))
        print(f"    {src.name:<10} {n_post:>6} post-event labels, {n_img:>6} images")
        if n_post == 0:
            problems.append(f"{src.name}: no *_post_disaster.json under labels/")
        if n_img < n_post:
            problems.append(
                f"{src.name}: {n_post - n_img} labelled tiles have no image"
            )
    if not found_any:
        problems.append(
            "no <source>/images + <source>/labels pair found; the archive layout "
            "differs from what parse_annotations.py expects"
        )
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urls", type=Path, default=Path("configs/xbd_urls.txt"))
    ap.add_argument("--dest", type=Path, default=Path("data/raw"))
    ap.add_argument("--manifest", type=Path, default=Path("notes/dataset.md"))
    ap.add_argument(
        "--keep-archives",
        action="store_true",
        help="keep the tarballs after extraction (needs ~2.2x the disk)",
    )
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument(
        "--from-local",
        type=Path,
        nargs="+",
        default=None,
        help="ingest archives already on disk (e.g. downloaded by a browser) "
        "instead of fetching URLs — same hashing, extraction and verification",
    )
    ap.add_argument(
        "--consume",
        action="store_true",
        help="with --from-local, delete each archive after extracting it. Off by "
        "default: those are your files, not ours.",
    )
    ap.add_argument(
        "--allow-geotiff",
        action="store_true",
        help="proceed with the full GeoTIFF release (needs rasterio; see preflight)",
    )
    args = ap.parse_args()

    if args.verify_only:
        problems = verify_layout(args.dest)
        for p in problems:
            print(f"  ! {p}")
        return 1 if problems else 0

    if args.from_local:
        # A browser download is a perfectly good way to get the data. It just
        # skips the hashing, extraction and layout check, so route it back
        # through the same path. Note that Safari transparently gunzips, so a
        # .tar.gz can arrive as .tar — the recorded hash is of the file actually
        # used, which will not match the publisher's checksum for the .tar.gz.
        archives = []
        for path in args.from_local:
            if path.is_dir():
                archives += sorted(
                    p for p in path.iterdir() if p.suffix in {".tar", ".gz", ".tgz"}
                )
            elif path.exists():
                archives.append(path)
            else:
                print(f"warning: {path} does not exist, skipping", file=sys.stderr)
        if not archives:
            print("no archives found to ingest", file=sys.stderr)
            return 1

        args.dest.mkdir(parents=True, exist_ok=True)
        total_gb = sum(a.stat().st_size for a in archives) / 1024**3
        available = free_gb(args.dest)
        print(f"{len(archives)} local archive(s), {total_gb:.1f} GB")
        print(f"extracted contents need ~{total_gb * 1.1:.1f} GB, "
              f"{available:.1f} GB free at {args.dest}")
        if available < total_gb * 1.1:
            print("\nNot enough disk to extract these.", file=sys.stderr)
            return 1

        entries = []
        for archive in archives:
            print(f"\n--> {archive.name}")
            size = archive.stat().st_size
            print(f"    {size / 1024**3:.2f} GB, hashing…")
            entries.append(
                {"file": archive.name, "bytes": size, "sha256": sha256(archive)}
            )
            extract(archive, args.dest)
            if args.consume:
                archive.unlink()
                print("    archive deleted (--consume)")

        print("\nverifying layout")
        problems = verify_layout(args.dest)
        for p in problems:
            print(f"  ! {p}")
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with args.manifest.open("a") as f:
            f.write(
                "\n".join(
                    [
                        f"\n## Local ingest {stamp}\n",
                        "```json",
                        json.dumps(entries, indent=2),
                        "```\n",
                    ]
                )
            )
        print(f"\nrecorded {len(entries)} archive(s) in {args.manifest}")
        return 1 if problems else 0

    if not args.urls.exists():
        args.urls.parent.mkdir(parents=True, exist_ok=True)
        args.urls.write_text(
            "# One xView2 download URL per line. Blank lines and # comments ignored.\n"
            "# Get these by logging in at https://xview2.org and copying the\n"
            "# download links. They are signed and expire — re-copy if a 403 appears.\n"
            "#\n"
            "# Start with the challenge training set only. tier3 doubles the disk\n"
            "# for data this project does not need.\n"
        )
        print(f"created {args.urls} — paste your download links into it, then re-run.")
        return 1

    urls = [
        line.strip()
        for line in args.urls.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not urls:
        print(f"no URLs in {args.urls}", file=sys.stderr)
        return 1

    jobs = plan(urls)
    preflight(jobs, args.allow_geotiff)

    archive_gb = guess_archive_gb([u[0] for _, u in jobs])
    needed = archive_gb if args.keep_archives else archive_gb * 1.2
    needed += archive_gb * 1.2  # extracted contents
    available = free_gb(args.dest)
    parts = sum(len(u) for _, u in jobs)
    print(f"{len(jobs)} archive(s) from {parts} URL(s), ~{archive_gb:.0f} GB compressed")
    print(f"need ~{needed:.0f} GB, have {available:.0f} GB free at {args.dest}")
    if available < needed:
        print(
            f"\nNot enough disk. Either drop tier3 from the URL list, or run this "
            f"on the pod's network volume instead of the laptop.",
            file=sys.stderr,
        )
        return 1

    entries = []
    for name, part_urls in jobs:
        downloaded = [download(u, args.dest) for u in part_urls]
        archive = (
            downloaded[0]
            if len(downloaded) == 1
            else join_parts(downloaded, args.dest / name)
        )
        size = archive.stat().st_size
        print(f"    {size / 1024**3:.2f} GB, hashing…")
        digest = sha256(archive)
        entries.append({"file": archive.name, "bytes": size, "sha256": digest})
        extract(archive, args.dest)
        if not args.keep_archives:
            archive.unlink()
            print("    archive removed (pass --keep-archives to retain)")

    print("\nverifying layout")
    problems = verify_layout(args.dest)
    for p in problems:
        print(f"  ! {p}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    block = [
        f"\n## Download {stamp}\n",
        "```json",
        json.dumps(entries, indent=2),
        "```\n",
    ]
    with args.manifest.open("a") as f:
        f.write("\n".join(block))
    print(f"\nrecorded {len(entries)} archive(s) in {args.manifest}")
    print("\nNow paste the license text from xview2.org into notes/dataset.md.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
