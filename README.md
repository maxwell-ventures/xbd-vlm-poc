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
python scripts/smoke_test.py          # verifies schema + metrics, no data needed
```

Then, once xBD is downloaded to `data/raw/`:

```bash
python scripts/parse_annotations.py --raw data/raw --out data/labels.csv
python scripts/build_chips.py --labels data/labels.csv --chips data/chips --pre
python scripts/split.py --labels data/labels.csv --out configs/split.json
python scripts/build_dataset.py --per-class 1500 --out data/processed
```

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

xBD is redistributed under its own terms — see the xView2 site. Nothing from
`data/raw/` is committed here. Record the dataset version and license text in
`notes/dataset.md` at download time.
