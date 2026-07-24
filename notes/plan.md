# Plan

## Blocking, do first

- [ ] Register at xview2.org and accept the dataset terms. **Access is gated
      behind an account** — do not discover this on Saturday morning.
- [ ] Record the dataset version and license text in `notes/dataset.md`.
- [ ] Create the RunPod account and a ~150 GB network volume.

## Compute

Local machine has 35 GB free — not enough for xBD. So the dataset lives on the
pod's network volume, and prep runs there too. The laptop holds the repo.

| what | where | why |
|---|---|---|
| xBD raw, chips, processed jsonl | pod network volume | disk |
| parse / chip / split / build_dataset | pod (CPU) | data is there |
| training | pod (GPU, attached only while working) | VRAM |
| predict | pod (GPU) | VRAM |
| evaluate, writeup, demo | laptop | no GPU needed |

Card: 24 GB (A10G / L4 / RTX 4090) at ~$0.30–0.50/hr is enough for 3B QLoRA.
Kill the pod between sessions; the volume persists at ~$0.07/GB/month.

Expected total: **$15–30**.

## Sequence

### Session 1 — data
- [ ] Download one tier of xBD to the volume (`train`, not all of it)
- [ ] `parse_annotations.py` → labels.csv; sanity-check the grade distribution
      against the ~80% no-damage expectation
- [ ] `build_chips.py --pre`; eyeball 20 chips before trusting any of them
- [ ] `split.py`; **resolve every coverage warning** before continuing

### Session 2 — evaluation harness and baseline
- [ ] `build_dataset.py` (post-only, `--tag post`)
- [ ] `predict.py` with no adapter → `outputs/eval/base_post.jsonl`
- [ ] `evaluate.py` → baseline numbers. Read the failure examples.
- [ ] Record the baseline in `notes/results.md` before training anything.

The baseline is the argument. If the zero-shot unparseable rate is above ~40%,
fix the prompt before training — a strawman baseline inflates the delta with
format compliance rather than assessment skill.

### Session 3 — run 1 (post-only)
- [ ] LoRA config: rank 16, alpha 32, LR ~1e-4, 2 epochs, cosine + warmup
- [ ] Target attention + MLP projections of the language model; also the
      multimodal projector. Vision encoder frozen.
- [ ] Write `run_config.json` (template fingerprint, dataset sha256, all
      hyperparameters) into the adapter directory
- [ ] `predict.py --adapter` → `evaluate.py --baseline`

### Session 4 — run 2 (pre + post)
- [ ] `build_dataset.py --pre --tag prepost`, retrain, re-evaluate
- [ ] The pre-vs-post delta is the ablation worth talking about

### Session 5 — demo and writeup
- [ ] Fixed image, swap event type → show assessment change
- [ ] Fixed image, swap days-since → show priority change
- [ ] Same two prompts against the base model for contrast
- [ ] `notes/results.md`: method, numbers, failure modes, limitations

## Things to be able to explain cold

Check these off as they become true. This is the actual deliverable.

- [ ] What LoRA does — the low-rank decomposition, what `alpha` scales, which
      modules were targeted and why
- [ ] Why the vision encoder is frozen, and why that is a bet on nadir imagery
- [ ] What happens to an image between pixels and the language model's tokens
- [ ] Why the split is grouped by event and what random splitting would inflate
- [ ] Why balanced training but per-class reporting, and why accuracy is
      reported against two denominators
- [ ] What QWK measures and why accuracy alone is misleading here
- [ ] What the unparseable rate is and why it belongs in the results table
- [ ] Why the zero-shot baseline gets the schema in its prompt
- [ ] Why the evidence field is format learning, not grounded reasoning
