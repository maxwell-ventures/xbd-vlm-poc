# Disaster damage assessment via VLM fine-tuning

> **A learning experiment, not a product.** The goal is hands-on fluency with
> vision-language-model adaptation — the pipeline, the trade-offs, and how to
> measure whether fine-tuning actually helped. It is not a production system, it
> is not deployable, and it does not try to beat published xView2 leaderboard
> scores. Where a choice trades polish for something worth understanding, this
> writes the choice and its cost down rather than hiding it.

Fine-tunes an open-weight VLM to grade structural damage in post-disaster
satellite imagery, **conditioned on stated event context**, and emits a
machine-parseable assessment with cited evidence and a triage priority.

The result that matters is the **delta against the same model zero-shot** — not
absolute accuracy.

---

## Why this, and why a VLM

A convolutional classifier already wins on raw accuracy over xBD, so accuracy is
not the point. The point is to build fluency with the three things a VLM can do
that a classifier structurally cannot — and to learn where each one is real
versus where it only looks real:

1. **Context conditioning.** A collapsed roof means something different after a
   hurricane than after a wildfire. Event type and days-since-event enter as
   *prompt text* and change the output.
2. **Natural-language output with cited evidence**, so an assessment is
   reviewable by a human responder rather than a bare class index.
3. **A meaningful zero-shot floor.** A pretrained VLM already answers, so there
   is a real baseline to measure against.

Every one of those has a catch this project is deliberate about — the evidence
is templated, the priority is a fixed rule, and the conditioning is built into
the targets rather than discovered. Being precise about *what the artifact does
and does not prove* is the actual deliverable. The honest limitations are
[collected below](#honest-limitations) and threaded through the docs.

---

## Status

| phase | state |
|---|---|
| Acquire · Parse · Split · Chip · Build dataset | ✅ complete |
| Zero-shot baseline | ✅ measured — **QWK 0.012**, 0% unparseable |
| Train (LoRA) | ⬜ not started |
| Evaluate delta · Demo | ⬜ not started |

**304,370** graded buildings across **19 disaster events** parsed; **8,310**
chipped; **5,921 / 796 / 1,593** train/val/test examples built, class-balanced.

The zero-shot base model answered `minor-damage` on 1,578 of 1,593 test chips
and never once predicted `no-damage` or `destroyed` — a constant predictor with
perfect format compliance. Full numbers and the confusion matrix:
[notes/results.md](notes/results.md).

---

## Documentation

| doc | what's in it |
|---|---|
| [notes/architecture.md](notes/architecture.md) | the pipeline stage by stage, with diagrams: how a polygon becomes a training example, why the split is grouped by event, what each metric catches |
| [notes/decisions.md](notes/decisions.md) | every non-obvious choice and its cost, in prose — the file to reread before explaining the project |
| [notes/results.md](notes/results.md) | measured numbers, starting with the zero-shot baseline |
| [notes/benchmark-context.md](notes/benchmark-context.md) | the xView2 competition: SOTA scores, and why ours are not directly comparable |
| [notes/plan.md](notes/plan.md) | phased sequence and the things to be able to explain cold |
| [notes/dataset.md](notes/dataset.md) | xBD provenance, licence reasoning, archive hashes |
| [notes/pod-setup.md](notes/pod-setup.md) | the rented-GPU workflow and cost sanity check |

---

## Where it runs

Data prep, dataset building and scoring are CPU-only and run on a laptop.
Training and full-precision inference run on a rented GPU. The repo is the unit
of transfer — `git push` locally, `git pull` on the pod, and **never edit code
on the pod**. The baseline above was measured locally on Apple silicon via MLX;
see the [precision caveat](notes/results.md) before comparing it to a GPU-trained
model.

---

## Quickstart

```bash
pip install -r requirements.txt
git config core.hooksPath .githooks   # blocks secrets; per clone, not inherited
python scripts/smoke_test.py          # verifies schema + metrics, no data needed
```

Data pipeline (see [notes/pod-setup.md](notes/pod-setup.md) for where to run it):

```bash
cp .env.example .env                  # fill in HF_TOKEN; never commit this
python scripts/download_xbd.py        # creates configs/xbd_urls.txt on first run
# paste xView2 download links into that file, then:
python scripts/download_xbd.py --dest data/raw

python scripts/parse_annotations.py --raw data/raw --out data/labels.csv
python scripts/split.py              --labels data/labels.csv --out configs/split.json
python scripts/build_chips.py        --chips data/chips --pre --per-class 1500
python scripts/build_dataset.py      --per-class 1500 --tag post --out data/processed
```

Baseline on Apple silicon, no GPU rental:

```bash
python3 -m venv .venv && .venv/bin/pip install mlx-vlm
.venv/bin/python scripts/predict_mlx.py \
    --data data/processed/test_post.jsonl \
    --out outputs/eval/base_post.jsonl --resume
```

Score anywhere:

```bash
python scripts/evaluate.py --pred outputs/eval/tuned.jsonl \
    --baseline outputs/eval/base.jsonl --out outputs/eval/run1.json
```

Full stage-by-stage detail: [notes/architecture.md](notes/architecture.md).

---

## Experiments

| run | images | question |
|---|---|---|
| baseline | post only | what does the untuned model do? |
| run 1 | post only | does LoRA on the language model teach the mapping? |
| run 2 | **pre + post** | how much of the remaining error is change detection? |

The no-damage/minor boundary is fundamentally a change-detection problem and the
human annotators had both images. Run 2 gives a real ablation with a number
attached rather than a guess.

---

## Honest limitations

Not disclaimers — these are the point. Knowing exactly which of them applies is
what the experiment is for.

- **Evidence text is templated.** A correct evidence sentence is not proof the
  model looked at the right pixels.
- **Priority is a stated rubric**, a deterministic function of grade and
  days-since-event. The model applies a rule it was taught, it does not infer
  urgency.
- **The test set is class-balanced**, so overall accuracy is not a deployment
  figure over a real disaster's building stock.
- **The split deviates from the official xView2 split** — deliberately, because
  the official one leaks across a boundary this project cares about.
- **Earthquake cannot be tested cross-event.** xBD has one earthquake.
- **`volcanic-eruption` has 6 test examples.** Do not quote a number for it.
- **Event dates in `events.py` are approximate** and drive `days_since_event`.
- **Context conditioning is by construction**, built into the targets, not
  discovered by the model.

---

## License

xBD is **CC BY-NC-SA 4.0** — non-commercial, and adaptations inherit the licence,
which covers the generated chips, the processed training files, and
(conservatively) the trained adapter. Nothing derived from `data/raw/` is
committed here. Reasoning and attribution text: [notes/dataset.md](notes/dataset.md).

The **code** — `scripts/`, `xbd_vlm/`, `configs/`, `.githooks/` and the docs — is
released under the [MIT License](LICENSE). That licence covers the code only. It
does **not** extend to the xBD dataset or anything derived from it (the generated
chips, the processed training files, the trained adapter), which remain
CC BY-NC-SA 4.0 and are not distributed in this repo.
