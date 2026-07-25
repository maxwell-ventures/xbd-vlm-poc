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

## Run 3 — the 7B arm, and the full 2×3 grid

The 7B arm mirrors the 3B arm exactly: zero-shot baseline, LoRA post-only, LoRA
pre+post, same config, same test buildings. `Qwen2.5-VL-7B-Instruct`, bf16 on an
A40, run overnight as one chain (~10 h). Adapters in `outputs/adapters/*_7b`,
template fingerprint matches the 3B runs.

### The headline grid (QWK)

```
              zero-shot    tuned post    tuned pre+post
   3B            0.009         0.546          0.570
   7B            0.460         0.556          0.604
```

### macro-F1 (same cells)

```
              zero-shot    tuned post    tuned pre+post
   3B            0.109         0.503          0.528
   7B            0.295         0.503          0.503
```

### Three findings

**1. Capacity transforms the zero-shot floor. This is the biggest single effect
in the grid.** The 3B base is a degenerate constant predictor (QWK 0.009). The 7B
base already grades, zero-shot, at 0.460 — nearly the level the 3B model reaches
only *after* fine-tuning (0.546). Its zero-shot destroyed recall is 0.806. So most
of the celebrated 3B fine-tuning delta (+0.537) was a small model catching up to
where the large model already sat out of the box.

**2. Fine-tuning is the great equalizer, on post-only.** 3B tuned post (0.546) and
7B tuned post (0.556) are effectively tied. If you are going to fine-tune anyway,
the capacity advantage on the single-image task nearly vanishes. This is the first
of the three scenarios laid out when we decided to measure a 7B baseline: capacity
lifts the base, adaptation closes the gap.

**3. But capacity and information compound at the top.** The best cell is 7B +
pre+post (0.604), clearly above every other. And the before-image helped 7B *more*
than it helped 3B (7B +0.048, 3B +0.024): the more capable model exploits the extra
input better. So "capacity or information?" has no single winner. The best result
needs both, and they are synergistic rather than substitutable.

### The honest nuance

On macro-F1 the four tuned cells are all ~0.50, essentially tied. 7B's QWK edge
therefore comes from getting the ordinal *distances* right (fewer far-off errors),
not from better per-class balance. The win is ordinal calibration, not solving the
hard classes.

### The through-line failure: nothing fixed minor/major

Across every tuned run — 3B post, 3B pre+post, 7B post, 7B pre+post — minor-damage
recall stays poor and collapses into major-damage. At 7B pre+post it is 0.155, the
worst of all. Three separate levers were pulled at it: fine-tuning, the before-image,
and doubling model capacity. None moved it. The minor/major boundary is genuinely
hard from overhead imagery, mirroring the human-annotator disagreement the brief
predicted. A clean, honest negative result.

### Cost

Whole project: ~$8.33 of GPU (23.8 h on one A40, ephemeral, no volume). Pod
terminated after pulling all artifacts.

## The context-conditioning demo

The brief's centerpiece: hold one post-event image fixed, vary only the stated
context, and see whether the assessment moves. Run locally (Apple silicon, 3B
base vs tuned) on one Hurricane Michael building (true grade major-damage).
`scripts/demo_conditioning.py`, results in `outputs/eval/demo_conditioning.json`,
figure `outputs/figures/10_context_conditioning.png`.

**Base model:** predicts minor-damage / moderate on all seven variants. It parrots
event-appropriate words in the evidence prose (say "flood" and it mentions water),
but the grade and priority never move. Context-blind where it counts.

**Tuned model — the assessment tracks the text:**

- *Event type → evidence clause* (the conditioning we built): hurricane gives
  "debris aligned along the prevailing wind direction", flood gives "standing
  water surrounding the structure", wildfire gives "charring on adjacent parcels",
  earthquake gives "adjacent structures showing comparable collapse". Same pixels.
- *wildfire → no-damage.* Told it is a wildfire, the model reasons that the
  structure shows no charring and downgrades. A coherent conditioning behaviour.
- *days-since → priority* (the rule we built): days=3 gives moderate, larger days
  give critical.

**The honest finding — the demo exposes its own seams.** Two of them:

1. The base model already adapts its evidence *prose* to the stated event, so
   "the evidence text changed" is not by itself proof of learned conditioning. The
   real, classifier-impossible tell is that the tuned model's **grade and priority**
   move with context while the base model's never do.
2. Varying days-since shifts not just the priority but the **damage grade**
   (minor at 3 days, major at 45 and 200). Days cannot change a grade on identical
   pixels. This is a **spurious coupling** the model absorbed from the templated,
   rule-based targets — a fingerprint of conditioning that was trained in rather
   than reasoned. It is exactly the limitation the writeup should show, not hide:
   the conditioning is real and demonstrable, and it is not grounded reasoning.

This is a single example, so the specific shifts are illustrative, not measured.
The qualitative contrast (base flat, tuned responsive) is robust; the days→grade
coupling is a flagged observation worth a controlled follow-up.

### The demo is circular — and the measurement that isn't

The demo has a fatal flaw as evidence, worth stating bluntly: it cannot separate
the model *using* context from the model *parroting* context. Feed it "flood" and
it emits the memorised flood clause regardless of pixels. "Give the model wrong
information and it predicts wrong information" is trivially true and proves
nothing. Our templated targets *guaranteed* this: the only thing the model could
learn was event-word → canned-clause, never grounding.

The non-circular question is whether the **correct** context makes grading
**better** than no context. We have the data for it: 15% of examples carry event
type masked to "unknown." Splitting the tuned test predictions on that flag:

| model | correct context (QWK) | masked (QWK) | gap |
|---|---|---|---|
| 3B post | 0.549 | 0.535 | +0.014 |
| 3B pre+post | 0.589 | 0.490 | +0.099 |
| 7B post | 0.588 | 0.399 | +0.189 |
| 7B pre+post | 0.631 | 0.477 | +0.154 |

Correct context helps grading in every run, and the effect is large at 7B. This
is a real result and it is not circular: identical images, the only change is
whether the true event type was supplied.

**But it is confounded, and we have caught the confound in the act.** The
improvement has two possible sources this split cannot separate:

1. *Disambiguation (good):* context as a prior that resolves genuinely ambiguous
   pixels.
2. *Base-rate shortcut (bad):* the model learned P(grade | event) from the
   training mix and uses the event word as a cheat sheet, independent of pixels.

The days→grade coupling in the demo above is direct evidence that at least some of
the context-dependence is the shortcut kind. So the +0.19 at 7B is part real help,
part a more capable model exploiting a spurious correlation, ratio unknown.

### Named follow-up: disambiguation vs shortcut

One controlled run separates the two. Feed *wrong but plausible* context and
measure whether it hurts, restricted to genuinely ambiguous chips:

- If correct context helps **and** wrong context hurts **specifically on the
  ambiguous cases**, that is disambiguation — the valuable, classifier-impossible
  behaviour.
- If wrong context flips confident, correct calls, that is the shortcut.

Until that run exists, the honest claim is narrow: the tuned model incorporates
prompt context into its grade (a classifier cannot), correct context improves
grading on aggregate, and an unknown share of that improvement is a spurious
base-rate shortcut rather than grounded reasoning.
