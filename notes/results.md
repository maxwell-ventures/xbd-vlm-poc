# Results

## Baseline — zero-shot, post-event chip only

**Recorded before any training run.** Scored once on the held-out test set.

| | |
|---|---|
| model | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` |
| backend | MLX on Apple M1 Pro, greedy decoding |
| data | `data/processed/test_post.jsonl`, sha256 `8f3edbb38b2f` |
| template fingerprint | `897e142cced62d87` |
| examples | 1593 (balanced, ~398 per class) |

```
accuracy (all)      0.252        unparseable        0.0%
macro F1            0.104        ordinal MAE        0.992
QWK                 0.012        adjacent errors   50.5%
priority accuracy   0.250        distant errors    24.4%
```

| class | precision | recall | F1 | n |
|---|---|---|---|---|
| no-damage | 0.000 | 0.000 | 0.000 | 398 |
| minor-damage | 0.252 | 0.997 | 0.403 | 399 |
| major-damage | 0.200 | 0.008 | 0.014 | 399 |
| destroyed | 0.000 | 0.000 | 0.000 | 397 |

```
confusion (rows=true, cols=pred)
                no-damage  minor-dam  major-dam  destroyed
no-damage               0        397          1          0
minor-damage            0        398          1          0
major-damage            0        396          3          0
destroyed               0        387         10          0
```

### Reading

The model answered `minor-damage` on 1578 of 1593 chips and `major-damage` on
the other 15. It never once predicted `no-damage` or `destroyed` — two columns
of the confusion matrix are empty.

QWK of 0.012 is chance agreement. Accuracy of 0.252 is exactly the frequency of
the class it always names, which is what a constant predictor scores by
construction, and slightly *below* the 0.25 that uniform random guessing would
achieve on four balanced classes.

**Format compliance was perfect: 0 unparseable outputs in 1593.** The schema
instruction in the prompt does its job. This matters for the headline — every
point of improvement from fine-tuning will be assessment skill, with no
contribution from the tuned model merely learning to emit the right shape.

The evidence text is fluent and invented. A destroyed building drew *"the
structure shows signs of slight damage, with visible cracks and deformation in
the walls and roof."* Confident, well-formed, wrong.

### Per-event-type numbers are misleading on their own

| event type | n | accuracy | macro F1 | QWK |
|---|---|---|---|---|
| hurricane | 660 | 0.412 | 0.148 | 0.007 |
| tornado | 461 | 0.262 | 0.111 | 0.010 |
| wildfire | 256 | 0.023 | 0.012 | 0.013 |
| tsunami | 120 | 0.000 | 0.000 | −0.006 |
| flood | 90 | 0.022 | 0.011 | 0.000 |
| volcanic-eruption | 6 | 0.000 | 0.000 | 0.000 |

Hurricane does not look better because the model understands hurricanes. The
test set is balanced *overall*, not within each event, so a constant
`minor-damage` predictor scores wherever minor damage happens to be common. QWK
is ~0 everywhere, which is the honest summary. Report accuracy per slice only
alongside that slice's class distribution.

Volcanic-eruption has 6 test examples, because `guatemala-volcano` is the only
held-out volcano event and has just 682 buildings total. Do not quote a number
for that slice.

---

## Baseline — bf16 on GPU (the reported baseline)

Re-measured on an A40 with `scripts/predict.py` in **bf16**, the same precision
and backend as the tuned model will use. This is the number the delta is measured
against; the laptop MLX run above was the cheap dress rehearsal.

| metric | laptop 4-bit MLX | **pod bf16** |
|---|---|---|
| accuracy (all) | 0.252 | **0.254** |
| macro F1 | 0.104 | **0.109** |
| QWK | 0.012 | **0.009** |
| ordinal MAE | 0.992 | **0.995** |
| unparseable | 0.0% | **0.0%** |

```
confusion (rows=true, cols=pred)   [bf16]
                no-damage  minor-dam  major-dam  destroyed
no-damage               0        395          0          3
minor-damage            0        398          0          1
major-damage            0        399          0          0
destroyed               0        390          0          7
```

**The precision mismatch was a non-issue.** The two baselines agree to within
noise: the base model is a degenerate constant predictor in both precisions,
QWK ≈ 0 either way. Quantization was never what produced the flat baseline. The
bf16 run shifts a handful of predictions into `destroyed` but the picture is
unchanged — it still never predicts `no-damage` and still funnels ~98% of chips
into `minor-damage`.

File: `outputs/eval/base_post_gpu.json`, generations in `base_post_gpu.jsonl`.

---

## Run 1 — LoRA, post-only

Adapter: `outputs/adapters/run1_post` (rank 16, α 32, lr 1e-4, 2 epochs, best
eval loss 0.115 at step ~550 of 740). Vision encoder + projector frozen. Scored
against the bf16 baseline on the identical test set.

### The headline delta

| metric | base | **tuned** | delta |
|---|---|---|---|
| QWK | 0.009 | **0.546** | **+0.537** |
| macro F1 | 0.109 | **0.503** | +0.394 |
| accuracy (all) | 0.254 | **0.517** | +0.262 |
| ordinal MAE (lower) | 0.995 | **0.629** | −0.366 |
| unparseable | 0.0% | **0.0%** | — |

QWK moves from chance to moderate–substantial agreement. The baseline was fair
(0% unparseable, same precision, same prompt), so none of this is format-
compliance inflation — it is assessment skill.

### Per-class recall — the model went from degenerate to discriminating

| grade | base | tuned |
|---|---|---|
| no-damage | 0.000 | 0.271 |
| minor-damage | 0.997 | 0.313 |
| major-damage | 0.000 | 0.722 |
| destroyed | 0.018 | 0.761 |

```
confusion (rows=true, cols=pred)   [tuned]
                no-damage  minor-dam  major-dam  destroyed
no-damage             108        142         98         50
minor-damage            0        125        261         13
major-damage           10         47        288         54
destroyed               4          3         88        302
```

The minor-damage recall *drop* is not regression: the base only "won" that class
by labelling everything minor. The real weakness is real and expected — **261 of
399 minor-damage chips are called major-damage.** That is the minor/major
boundary the brief flagged as hardest, and the one predicted to suffer under
post-only chipping because it is fundamentally change detection. no-damage recall
(0.271) is weak for the same reason: an intact roof is ambiguous without a
"before" image. Both are the explicit motivation for run 2.

### Slices worth noting

- **Context withheld barely hurts** (QWK 0.549 shown vs 0.535 withheld). The model
  is not over-reliant on the event-type prompt field — the healthy result.
- **Per event type:** wildfire QWK 0.796, tornado 0.656 (sharp, localised damage
  reads well); hurricane 0.141 and tsunami 0.114 are weakest. Do not quote the
  volcanic-eruption slice — n=6.

## Run 2 — LoRA, pre + post

Adapter: `outputs/adapters/run2_prepost` (identical recipe to run 1, best eval
loss 0.115). The one variable changed from run 1 is the input: each example now
carries the pre-event image as well as the post-event image. Same 1593 test
buildings in the same order (selection is a uid-hash keyed on the split name,
independent of the `--pre` flag), so run 1 vs run 2 is a clean one-variable
ablation isolating the effect of the before-image.

### Run 1 → Run 2 (isolates the before-image)

| metric | run 1 (post) | run 2 (pre+post) | delta |
|---|---|---|---|
| QWK | 0.546 | 0.570 | +0.024 |
| macro-F1 | 0.503 | 0.528 | +0.025 |
| accuracy | 0.517 | 0.529 | +0.012 |
| ordinal MAE (lower) | 0.629 | 0.615 | −0.014 |
| unparseable | 0.0% | 0.0% | — |

| grade | run 1 recall | run 2 recall | delta |
|---|---|---|---|
| no-damage | 0.271 | 0.422 | **+0.151** |
| minor-damage | 0.313 | 0.316 | +0.003 |
| major-damage | 0.722 | 0.729 | +0.008 |
| destroyed | 0.761 | 0.647 | **−0.113** |

```
confusion (rows=true, cols=pred)   [run 2, pre+post]
                no-damage  minor-dam  major-dam  destroyed
no-damage             168        112         88         30
minor-damage           38        126        224         11
major-damage           30         53        291         25
destroyed              11         17        112        257
```

### Read: the hypothesis was half right, with a surprise

The prediction was that the before-image would lift the no-damage and minor↔major
cases, because both are fundamentally change detection and a post-only crop cannot
do change detection.

- **Confirmed on no-damage** (+0.151 recall). An intact building is now
  identifiable because it looks the same before and after. This is the clean
  change-detection win.
- **Wrong on the minor/major boundary.** Minor-damage recall moved +0.003, i.e.
  not at all. Minor-damage is still called major-damage 224 times (was 261 in run
  1 — barely dented). The boundary I was most confident the before-image would fix
  is the one that barely moved.
- **Unpredicted regression on destroyed** (−0.113 recall). Adding the before-image
  *hurt* the class run 1 handled best. destroyed→major misclassifications rose 88
  → 112. Working theory (unverified): for a flattened building the pre-image shows
  an intact structure, and reconciling "was a house, now a slab" pulls some
  predictions toward major instead of destroyed.

**Net:** the second image redistributed errors more than it reduced them. Large
no-damage gain, offsetting destroyed loss, minor/major untouched, small positive
aggregate (QWK +0.024). A +0.024 aggregate on 1593 examples is not decisive on its
own; the per-class redistribution is the real, directional signal.

This is a more useful result than a clean win: a prediction partly confirmed,
partly wrong on the case it was most sure of, plus an unexpected regression to
explain.

## Run 3 — LoRA, 7B (planned)

*Isolates model capacity: change only the base model (3B → 7B), holding the data
condition fixed at the best 3B run. Tests whether raw capacity does what the
before-image could not, especially on the stuck minor/major boundary.*
