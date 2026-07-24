# Dataset provenance

xBD / xView2 Challenge dataset. Nothing from `data/raw/` is committed to this
repo; this file is the record of exactly what was downloaded.

## License

> **TODO — paste the license text from xview2.org here at download time.**
>
> Record: the license name, the URL it was read from, the date accepted, and any
> restriction on redistribution or commercial use. This matters because the
> repo is going to be shown to people, and "I checked the terms" is only
> credible if the terms are written down.

## Version

| field | value |
|---|---|
| source | https://xview2.org |
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
