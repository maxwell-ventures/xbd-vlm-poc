# Dataset provenance

xBD / xView2 Challenge dataset. Nothing from `data/raw/` is committed to this
repo; this file is the record of exactly what was downloaded.

## License

**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0)** — https://creativecommons.org/licenses/by-nc-sa/4.0

Recorded 2026-07-23 from xview2.org.

Three clauses, and all three reach this project:

**BY — attribution.** Any use of the imagery, including screenshots in a demo or
a writeup, carries the attribution below.

**NC — non-commercial only.** This is a portfolio and learning artifact, which is
fine. It also means the pipeline cannot be lifted into commercial use without
licensing the data separately from the source.

**SA — share-alike.** Adaptations inherit the licence. The per-building chips in
`data/chips/` and the conversation files in `data/processed/` are adaptations of
xBD. Both are gitignored, so nothing is being distributed — but if they ever
were, they would have to ship under CC BY-NC-SA 4.0.

### The adapter

Whether fine-tuned weights are a derivative work of their training data is
legally unsettled. The conservative position, adopted here:

- The trained adapter is treated as **CC BY-NC-SA 4.0**, same as the data.
- If it is published (e.g. to the Hugging Face Hub), it is labelled NC, with a
  pointer to this file explaining why.
- It is not offered for commercial use.

This is a stance, not a legal opinion. It is written down so the choice is
visible rather than accidental.

### Code

The code in this repo is independent of the dataset and is **not** covered by
CC BY-NC-SA. Pick a licence for it explicitly (MIT or Apache-2.0 are the
conventional choices) and add a `LICENSE` file, so it is unambiguous that the
permissive licence covers `scripts/` and `xbd_vlm/` and **not** `data/` or any
published adapter.

### Attribution text

> Building damage annotations and imagery from the xBD dataset (xView2
> Challenge), licensed CC BY-NC-SA 4.0.
>
> Gupta et al., *Creating xBD: A Dataset for Assessing Building Damage from
> Satellite Imagery*, CVPR Workshops, 2019.

*(Verify the exact citation against the paper before using it in a writeup.)*

## Version

| field | value |
|---|---|
| source | https://xview2.org |
| licence | CC BY-NC-SA 4.0 |
| subsets downloaded | *(train / tier3 / test / hold — record which)* |
| download date | *(fill in)* |
| accessed by account | *(fill in — no password here)* |

## Expected layout

`parse_annotations.py` walks `data/raw/*/labels/*_post_disaster.json` and expects:

```
data/raw/<source>/images/<disaster>_<tile>_pre_disaster.png
data/raw/<source>/images/<disaster>_<tile>_post_disaster.png
data/raw/<source>/labels/<disaster>_<tile>_pre_disaster.json
data/raw/<source>/labels/<disaster>_<tile>_post_disaster.json
```

Damage grades live on the **post**-event annotations, under
`features.xy[].properties.subtype`. The pre-event annotations carry building
footprints without grades, and are not read.

Verify after extraction:

```bash
python scripts/download_xbd.py --verify-only --dest data/raw
```

## Archive hashes

Appended automatically by `download_xbd.py` on each run.

## Download 2026-07-24 01:59 UTC

```json
[
  {
    "file": "train_images_labels_targets.tar.gz",
    "bytes": 8381754539,
    "sha256": "a5941b7a3e523eafc4aeaa740a1c83f1af6a18c894e7e8c62dd830a76921ecd4"
  }
]
```
