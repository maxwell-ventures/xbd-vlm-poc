# Disaster damage assessment via VLM fine-tuning

Fine-tunes an open-weight vision-language model to grade structural damage in
post-disaster satellite imagery, conditioned on stated event context. The
headline result is the delta against the same model zero-shot — not absolute
accuracy, which published CNN baselines on xBD already beat.

Status: **pipeline scaffolded, no data yet.** See [notes/plan.md](notes/plan.md).

## Layout

```
xbd_vlm/          shared library — the only code that touches the schema
  schema.py       output format, parser, triage rule
  prompts.py      prompt template + evidence templating (versioned, hashed)
  events.py       event type and date per disaster
  metrics.py      ordinal metrics: per-class, QWK, MAE, unparseable rate
scripts/
  download_xbd.py        xView2       -> data/raw + hashes in notes/dataset.md
  parse_annotations.py   raw GeoJSON  -> data/labels.csv
  build_chips.py         labels.csv   -> per-building crops
  split.py               labels.csv   -> configs/split.json (grouped by event)
  build_dataset.py       labels.csv   -> data/processed/*.jsonl + dataset card
  predict.py             jsonl        -> raw generations          [GPU]
  evaluate.py            generations  -> metrics table            [CPU]
  smoke_test.py          synthetic end-to-end check on the no-GPU path
configs/          split assignment, training configs
outputs/          adapters (gitignored), eval metrics (tracked)
notes/            plan, decisions, results
```

## Where things run

Data prep, dataset building and scoring are CPU-only and run on a laptop.
Training and inference run on a rented GPU. The repo is the unit of transfer:
`git push` locally, `git pull` on the pod. Never edit code on the pod.

## Quickstart

```bash
pip install -r requirements.txt
git config core.hooksPath .githooks   # blocks secrets from being committed
python scripts/smoke_test.py          # verifies schema + metrics, no data needed
```

`core.hooksPath` lives in `.git/config`, which is not cloned — run that line in
every clone, including on the pod.

Then, on the machine with the disk for it (see [notes/pod-setup.md](notes/pod-setup.md)):

```bash
cp .env.example .env                  # fill in HF_TOKEN; never commit this
python scripts/download_xbd.py        # creates configs/xbd_urls.txt on first run
# paste your xView2 download links into that file, then:
python scripts/download_xbd.py --dest data/raw

python scripts/parse_annotations.py --raw data/raw --out data/labels.csv
python scripts/split.py --labels data/labels.csv --out configs/split.json
python scripts/build_chips.py --chips data/chips --pre --per-class 1500
python scripts/build_dataset.py --per-class 1500 --out data/processed
```

Split runs **before** chipping: xBD has ~850k polygons and this project uses a
few thousand, so `build_chips.py` crops only the sampled subset. Selection is a
deterministic hash of the uid, so `build_dataset.py` independently arrives at
exactly the same rows. Keep the `--per-class` values in step between the two
(raising the cap later is purely additive — existing chips stay valid).

On the GPU box:

```bash
pip install -r requirements-gpu.txt
python scripts/predict.py --data data/processed/test.jsonl --out outputs/eval/base.jsonl
# ... train ...
python scripts/predict.py --data data/processed/test.jsonl \
    --adapter outputs/adapters/run1 --out outputs/eval/tuned.jsonl
```

Back on the laptop:

```bash
python scripts/evaluate.py --pred outputs/eval/tuned.jsonl \
    --baseline outputs/eval/base.jsonl --out outputs/eval/run1.json
```

## Output schema

```
DAMAGE: major-damage
EVIDENCE: Substantial roof failure exposing the structure's interior; burn scars and ash across the surrounding vegetation.
PRIORITY: high
```

`DAMAGE` uses xBD's own four subtype strings. `PRIORITY` is a **stated rubric**,
a deterministic function of grade and days-since-event — not learned judgement.
`EVIDENCE` is **templated** from grade and event type; it is format learning, not
grounded visual reasoning. Both caveats are load-bearing and are repeated in
[notes/decisions.md](notes/decisions.md) and belong in any writeup.

## Honest limitations, up front

- Evidence text is templated, so a correct evidence sentence is not proof the
  model looked at the right pixels.
- The test set is class-balanced, so overall accuracy is not a deployment figure.
- The split deviates from the official xView2 split (grouped by event instead of
  within event) because the official one leaks across a boundary this project
  cares about.
- Event dates in `events.py` are approximate and drive `days_since_event`.

## License

xBD is **CC BY-NC-SA 4.0**. Non-commercial only, and adaptations inherit the
licence — which covers the generated chips, the processed training files, and
(conservatively) the trained adapter. Nothing derived from `data/raw/` is
committed here. Full reasoning and attribution text: [notes/dataset.md](notes/dataset.md).

The code in `scripts/` and `xbd_vlm/` is separate from the dataset and is not
covered by that licence.
