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

*pending*

## Run 2 — LoRA, pre + post

*pending*
