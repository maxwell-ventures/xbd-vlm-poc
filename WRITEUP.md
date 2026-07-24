<!--
  DRAFT SKELETON. Practitioner-teaching voice, lean/narrative.
  Purpose of each section is in an HTML comment above it. Prose marked [DRAFT]
  is real candidate copy to react to; [PENDING] waits on run 2 or the demo.
  Nothing here is wired into the README yet. We link it once it's fleshed out.
-->

# Teaching a vision-language model to grade disaster damage, and proving it wasn't already faking it

<!-- HOOK: lead with the degenerate baseline. Dramatize "fluent != correct" in the
     first three sentences. This is the whole thesis in miniature. -->

[DRAFT] Here is a state-of-the-art vision-language model assessing a building that
a hurricane reduced to a slab:

> **DAMAGE:** minor-damage
> **EVIDENCE:** The structure shows signs of slight damage, with visible cracks and deformation in the walls and roof.
> **PRIORITY:** moderate

Fluent. Confident. It even cites evidence. It is also wrong: the building is
destroyed. And it gave essentially that same answer to 98% of the images I showed
it, across every kind of damage, scoring right at chance on the actual grading
task while sounding authoritative the whole way through.

This post is about the gap between sounding right and being right, and about the
unglamorous work of measuring your way across it. I fine-tuned that same 3B model
to grade post-disaster satellite imagery. The interesting part isn't the accuracy
number, which a plain convolutional classifier beats. The interesting part is how
you prove the fine-tuned model learned something real instead of just learning a
new way to sound plausible.

---

## What I built, in one paragraph

<!-- Orient the reader fast: task, model, method, the ONE framing that matters. -->

[DRAFT] The task: given a satellite chip of one building and a line of context
("this was a wildfire, four days ago"), output a damage grade on xBD's four-level
scale, the visual evidence for it, and a triage priority. The method: LoRA
fine-tuning of Qwen2.5-VL-3B on about 6k class-balanced examples, with the vision
encoder frozen. And the framing that governs everything else: the result is the
delta over the same model zero-shot, not the absolute accuracy. A randomly
initialised classifier has no zero-shot floor to improve on. A pretrained VLM
does, and that floor is what turns "did fine-tuning teach it anything?" into a
question you can actually measure.

> Full pipeline and stage-by-stage detail: [notes/architecture.md](notes/architecture.md).

---

## Lesson 1: a baseline you didn't work for is a baseline that flatters you

<!-- Core teaching beat. Most writeups measure against nothing or an unfair
     baseline. Show the two ways this one is fair, and that fairness was WORK. -->

[DRAFT] The delta is only worth as much as the baseline is honest, and there are
two easy ways to cheat it without noticing.

The first is to withhold your instructions from the baseline. A zero-shot model
that has never seen your output schema will ramble, and you will book a large
"improvement" that is really just the tuned model learning to emit three tidy
lines. So the exact schema goes in both prompts. The payoff was visible in the
numbers: the base model's unparseable rate came in at 0%. It followed the format
perfectly and still could not grade. That means every point of the delta is
assessment skill and none of it is formatting.

The second is to measure the two models in different numerical precision. [DRAFT:
the precision-mismatch story. I ran the first baseline locally in 4-bit because it
was free, then realised the tuned model would run in bf16 on a GPU. Comparing
across precisions would quietly blend the effect of fine-tuning with the effect of
quantization. I re-ran the baseline in bf16. It matched the 4-bit run to within
noise (QWK 0.012 versus 0.009), which killed the confound and, as a bonus,
confirmed the flat baseline was not a quantization artifact. The lesson: name the
confound and spend the ten minutes, even when you are fairly sure of the answer.]

---

## Lesson 2: on imbalanced ordinal data, accuracy lies

<!-- Metric literacy. Why accuracy is the wrong headline, what QWK buys, the
     always-"no-damage" illustration. -->

[DRAFT] xBD is about 77% "no-damage." A model that ignores the image and always
answers "no-damage" scores 77% accuracy and is useless. I pinned that intuition in
a unit test: an always-"no-damage" predictor scores 0.81 accuracy and 0.00 QWK on
a naturally distributed set. If a constant can max your headline metric, it is the
wrong headline.

So the reported metrics are chosen to resist degeneracy:

- **Quadratic weighted kappa (QWK)** rates ordinal agreement corrected for chance.
  A constant predictor scores 0 no matter how flattering its accuracy looks.
- **Macro-F1** collapses the moment a class is never predicted.
- **Ordinal MAE** treats the scale as ordered, so confusing minor with major costs
  less than confusing no-damage with destroyed. Those are different failures and
  deserve different weights.
- **Unparseable rate** is counted, never quietly dropped.

[DRAFT: one line on reporting accuracy against two denominators, one that charges
for parse failures and one that doesn't.]

---

## Lesson 3: your train/test split decides what your number even means

<!-- Leakage. Event-grouped vs random split. -->

[DRAFT] The obvious split, shuffle every building and take 80/10/20, leaks badly
here. Buildings from one satellite tile share a sensor, a sun angle, a local
architecture, and an annotator. Put near-duplicate neighbours on both sides of the
wall and your "accuracy" measures tile memorisation, not whether the model
generalises to the next disaster. So the split is grouped by whole disaster event:
train on Hurricane Harvey, test on Hurricane Michael. [DRAFT: note this
deliberately departs from the official xView2 split, and the cost it creates. Hold
out a whole event and you can hold out the only example of its type. Six of seven
event types end up testable across events. Earthquake cannot be, because xBD
contains exactly one.]

---

## Lesson 4: know what your artifact can't claim

<!-- The intellectual-honesty beat. Templated evidence, rule-based priority,
     by-construction conditioning. Own it. -->

[DRAFT] xBD hands you a polygon and a grade. It hands you no rationale text. So the
"EVIDENCE" field my model learns to produce is templated, composed from a
damage-grade clause and an event-type clause. That carries a consequence I want to
state plainly rather than bury. A correct-sounding evidence sentence is not proof
the model looked at the right pixels. The field teaches output format and an
association between context and vocabulary. It does not teach grounded visual
reasoning, and I don't claim it does. [DRAFT: the same honesty for PRIORITY, which
is a stated rule of grade and days-since-event rather than learned judgement, and
for the context conditioning, which is built into the targets by construction
rather than discovered by the model. Then the turn: naming this is the point. The
honest version is more useful to a reader than an oversold demo, and the real
upgrade, distilling rationales from a stronger VLM, is the obvious next
experiment.]

---

## Lesson 5: what actually gets adapted, and the bug the loss curve won't show you

<!-- Practitioner mechanics, kept lean. LoRA, frozen encoder (a bet), and the
     label-masking gotcha a dropping loss won't reveal. -->

[DRAFT] The LoRA adapters attach to the language model's attention and MLP
projections. That is 37M trainable parameters, 0.98% of the model. The vision
encoder stays frozen, and this is a bet rather than a fact. The wager is that the
encoder already represents rubble and burn scars as texture, and the adaptation we
need lives downstream, in the mapping from visual features to a grade. On
off-distribution nadir satellite imagery that bet is arguable, and unfreezing the
projector is a named follow-up.

[DRAFT: the label-masking gotcha as the transferable lesson. Only the assistant
turn should contribute to the loss; the prompt and the roughly 400 image tokens
have to be masked out. A dropping loss curve will not tell you whether you got
this right. I checked it by decoding exactly the tokens that carried gradient and
confirming they were the 38-token answer and nothing else. There is a companion
detail worth stating: training pads on the right, batched generation pads on the
left, and mixing the two corrupts output silently.]

---

## The result

<!-- Payoff. Lead with the delta table, then the degenerate->discriminating story. -->

[DRAFT] Post-only fine-tuning, scored against the fair bf16 baseline on the
held-out test set:

| metric | base | tuned | delta |
|---|---|---|---|
| QWK | 0.009 | **0.546** | **+0.537** |
| macro-F1 | 0.109 | 0.503 | +0.394 |
| accuracy | 0.254 | 0.517 | +0.262 |
| ordinal MAE (lower is better) | 0.995 | 0.629 | −0.366 |

[DRAFT: the story the table doesn't tell. The base predicted minor-damage for
almost everything. Per-class recall went from all zeros to genuinely
discriminating: destroyed 0.018 to 0.761, major 0.000 to 0.722. QWK moved from
chance to moderate-to-substantial agreement.]

---

## Turning on my own result

<!-- The move that separates this from a demo. Attack the weakness, show you
     predicted it, then the controlled experiment that tests the fix. -->

[DRAFT] The one class that got worse is the honest part. Minor-damage recall
dropped, but the base only "won" that class by calling everything minor. The real
weakness sits in the confusion matrix: 261 of 399 minor-damage chips get labelled
major-damage. That is the minor/major boundary, the place human annotators
disagree most, and I had flagged it as the likely failure before running anything.
Telling minor from major is fundamentally a change-detection problem, and a
post-event crop on its own cannot do change detection.

So the fix is a controlled experiment, not a guess. Give the model both the before
and the after image.

[PENDING run 2: the pre+post ablation. Report whether the second image lifts
exactly the minor/major and no-damage cases that failed here. This is the section
that makes the piece a real experiment rather than a single training run: same
recipe, one variable changed, a number attached.]

---

## The demo: context conditioning you can see

<!-- Visual centerpiece for a public writeup. Build before publishing. -->

[PENDING build: hold one image fixed, change only the stated event type, show the
assessment change. Then hold the event fixed and change days-since, show the
priority change. Base model beside tuned model. This is the single most shareable
artifact in the project and the strongest evidence the conditioning is real. Gate
publication on it.]

---

## What this does and doesn't prove, and what comes next

<!-- Collected honest limitations (the point, not disclaimers), then the series
     thread so a reader knows more is coming. -->

[DRAFT: collect the limitations already stated. Templated evidence. A balanced
test set, so accuracy is not a deployment figure. Conditioning built in by
construction. The one-earthquake gap. Frame them as the boundary of the claim, not
as apologies. Close on the series thread: this is one entry in a repeatable method
(pick a capability, build a fair baseline, measure the delta, stay honest about the
limits), and name the next one or two candidates.]

---

<!-- Footer: link the repo, architecture.md, decisions.md. Keep the "learning
     experiment, not a product" framing consistent with the README. -->
