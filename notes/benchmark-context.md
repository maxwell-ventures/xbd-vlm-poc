# Benchmark context: the xView2 Challenge

xBD was released for the **xView2 Challenge** (Defense Innovation Unit, 2019–2020),
so there is a documented competition to place this work against. This file records
what SOTA looks like and — the honest part — why our numbers are *not* directly
comparable to it. The one finding that does transfer cleanly is the most important:
minor-damage is the hardest class for everyone, not just for us.

## The competition and its metric

- 2,000+ submissions. Winners deployed to the 2020 California wildfires, coastal
  hurricanes, and the 2019–2020 Australia bushfires.
- **Score = 0.3 × localization F1 + 0.7 × damage F1**, on the official held-out
  (imbalanced) test set.
- **damage F1 = harmonic mean of the four per-class F1s** (no-damage, minor, major,
  destroyed), computed at pixel/object level.
- Crucially, the task **includes localization** — finding the building footprints,
  which we skipped by using ground-truth polygons.

## The score landscape

| solution | damage F1 | combined |
|---|---|---|
| Official baseline | 0.061 | 0.284 |
| DeepDamageNet (2024) | 0.587 | 0.664 |
| 1st place (Durnov) | ~0.77 | ~0.80 |
| Modern SOTA (ChangeMamba) | — | ~0.78 |

The winner: an ensemble of 12 U-Nets (four backbones × three seeds, siamese pre/post),
266% over baseline. Two things worth noting:

- The winner's damage F1 (~0.77) is inferred: combined ~0.80 with localization ~0.86
  implies 0.7 × damage ≈ 0.542, so damage F1 ≈ 0.77.
- The baseline's damage F1 was **0.061** — a real CNN, near-useless at classification
  before serious engineering. That rhymes with our degenerate zero-shot VLM baseline
  (QWK 0.009): strong architectures start near the floor on this task too.

## The finding that transfers: minor-damage is the hardest class for everyone

> "In the xView2 competition, top solutions showed minor-damage F1 scores ranging
> from **0.16 to 0.25**, making it the hardest damage class to classify."

Our minor-damage F1, for comparison (balanced test set, see caveats):

| our run | minor-damage F1 |
|---|---|
| 3B post | 0.349 |
| 3B pre+post | 0.356 |
| 7B post | 0.295 |
| 7B pre+post | 0.227 |

We land in the same territory, often above it. This is the validation of our
through-line result. "Nothing fixed minor/major" across four tuned runs was never
our model failing. It is a fundamental property of the problem that the best-funded
ensembles in a DoD-sponsored competition also hit. Minor damage is genuinely
ambiguous from overhead imagery, which is exactly what the four-level scale's own
annotators disagreed on most.

## Why our numbers are NOT directly comparable

We cannot claim a scoreboard position. The comparison is apples-to-oranges on at
least five axes:

1. **We used ground-truth polygons.** We skipped localization entirely — the 0.3
   weight, and arguably the harder half of their task.
2. **Different metric.** Harmonic-mean damage F1 vs our QWK and macro-F1.
3. **Balanced test set.** Ours is ~400/class; theirs is the real ~77%-no-damage
   distribution. Balancing inflates minor-class recall relative to theirs, which is
   *why* our minor F1 can sit above their 0.25 even though the task is identically
   hard.
4. **Different split.** We grouped by event (cross-event, harder); they used the
   official within-event split.
5. **Different problem shape.** A generative VLM emitting text and a rationale vs
   pixel-level segmentation.

## What this reference legitimately buys us

- **A picture of SOTA:** ~0.77 damage F1 with localization solved, via large CNN
  ensembles. Our project never aimed to compete with that, and does not.
- **A baseline sanity check:** their CNN baseline started at 0.061 damage F1; our
  zero-shot VLM started at QWK 0.009. Both near the floor before adaptation.
- **External validation of the minor/major finding:** the field tops out at
  0.16–0.25 minor F1, so our stuck minor class is the problem, not our budget.

The dishonest move would be to quote our balanced-test QWK next to their combined
0.80 and imply a ranking. We do not. The honest value is context and validation,
not a place in line.

## Sources

- [IBM — The xView2 AI Challenge](https://www.ibm.com/think/insights/the-xview2-ai-challenge)
- [xView2 scoring code (DIUx-xView)](https://github.com/DIUx-xView/xView2_scoring/blob/master/xview2_metrics.py)
- [1st place solution — Durnov](https://github.com/vdurnov/xview2_1st_place_solution)
- [DeepDamageNet, arXiv:2405.04800](https://arxiv.org/html/2405.04800v1)
- [xFBD: Focused Building Damage Dataset and Analysis, arXiv:2212.13876](https://arxiv.org/pdf/2212.13876)
- [2nd place write-up — Computer Vision Talks](https://computer-vision-talks.com/2020-01-xview2-solution-writeup/)

*Figures cited from public summaries; verify exact values against the primary
sources before quoting them in a published writeup.*
