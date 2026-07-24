# Decisions log

Every non-obvious choice, why it was made, and what it costs. This is the file
to reread before an interview — most of the good questions live here.

---

## Model: Qwen2.5-VL-3B-Instruct, not 7B

The brief says start at 7B. Dropped to 3B: the goal is pipeline fluency, and 3B
trains 2–3× faster on a cheaper card, which buys more iterations. It also runs
locally at 4-bit for the demo, so showing the result does not require renting a
GPU. 7B is a drop-in scale-up; nothing in the pipeline changes.

**Cost:** absolute numbers will be lower than a 7B run would give. Since the
claim is a base-vs-tuned delta measured on the same model, this does not
undermine the argument.

---

## Frozen: vision encoder. Trained: language model adapters.

The stated reason for freezing the encoder is that it already knows what rubble
and burn scars look like as texture. That is a **bet, not a fact** — Qwen's
encoder was pretrained on natural images and documents, and 0.3–0.8 m nadir
satellite imagery is well outside that distribution.

Hedge: also train the multimodal projector/merger. It is small, and it is
exactly the layer that has to bridge the domain gap.

**If results disappoint, this is the first thing to ablate.**

---

## Output schema: three key-value lines, not JSON

Fewer tokens on punctuation, and a malformed generation degrades to "one field
missing" rather than "the whole object failed", which makes the unparseable-rate
metric informative about *what* went wrong.

The parser tolerates markdown decoration and a small enumerated alias list
(`Major Damage` → `major-damage`). Anything beyond that counts as unparseable.
Tolerance is deliberate and bounded: too strict and the baseline is a strawman,
too loose and the format metric means nothing.

---

## EVIDENCE is templated — and that is a real limitation

xBD gives a polygon and an ordinal grade. No rationale text exists. So the
evidence sentence is composed from a grade clause and an event-type clause.

**What this teaches the model:** output format, and an association between the
stated event context and damage vocabulary.

**What it does not teach:** grounded visual reasoning. A correct evidence
sentence is not evidence that the model attended to the right region.

Alternatives considered:

| option | what it buys | why not now |
|---|---|---|
| Templated | free, deterministic, reproducible | rationale carries no information |
| Distilled from a stronger VLM | genuinely grounded, varied text | ~$25 and a second pipeline; low VLM-learning payoff |
| Drop the field | simplest | deletes the "why a VLM, not a classifier" argument |

Chosen: templated, stated loudly. Distillation is the named next step.

---

## Both context fields are made load-bearing on purpose

Risk: if `event_type` appears in every prompt and is perfectly readable off the
pixels, the model can ignore the text entirely. Then swapping event type at demo
time changes the output only because the prompt is now out of distribution —
that is brittleness dressed as conditioning.

Three countermeasures, built in:

1. **EVIDENCE phrasing is conditioned on event type** (`prompts.py`), so the
   field changes the correct target.
2. **PRIORITY is conditioned on `days_since_event`** (`schema.py`), which reaches
   the model *only* through prompt text and cannot be read off the image at all.
   This is the cleanest measurable conditioning signal in the project.
3. **Event type is withheld on 15% of training examples** (`UNKNOWN_EVENT_RATE`),
   so the field has to be read rather than assumed.

**Still by construction, not emergent.** The demo shows the model learned the
conditioning we built in. It does not show the model discovered that hurricanes
and wildfires differ.

---

## Split grouped by whole event, deviating from the official xView2 split

A random building-level split puts near-duplicate neighbours from the same tile
on both sides — same sensor, lighting, vernacular, annotator. Reported accuracy
then measures tile memorisation.

The official xView2 split is drawn within events, so it does not satisfy this.
The deviation is deliberate and must be stated in the writeup.

**Cost it creates:** 19 events over 7 types, unevenly. Holding out a whole event
can hold out the only instance of its type, turning the result into a zero-shot
transfer claim rather than a fine-tuning claim. `split.py` reports per-type
coverage and warns when this happens.

---

## Balanced training *and* balanced test

xBD is ~80% no-damage. Training on that distribution yields a model that answers
"no-damage" and scores 80%. Hence per-class caps.

The test set is capped the same way, which means **reported accuracy is not a
deployment figure**. It is the right choice for measuring per-class skill on rare
classes with enough samples to support the number. The writeup must not quote
overall accuracy as operational performance.

`smoke_test.py` pins this: an always-"no-damage" model scores 0.81 accuracy and
0.00 QWK on a naturally distributed set. That contrast is why QWK and macro-F1
are reported alongside accuracy.

---

## Chipping: adaptive window, target centred, no outline

Crop side = larger bbox dimension × 3.0, clamped to [128, 512] px, resized to
448 (16 × 28, matching Qwen's patch grid).

- **Adaptive vs fixed extent:** adaptive guarantees the building fits, but because
  every chip is resized to 448 the apparent scale varies, so the model cannot
  read absolute building size. `--mode fixed` preserves scale and clips large
  structures. Cropping the subject is the worse failure. Worth ablating.
- **No polygon outline by default:** drawing the target's outline removes "which
  building?" ambiguity, but paints a non-photographic cue the model will learn —
  a cue that does not exist at inference on unlabelled imagery. Available behind
  `--outline` as an ablation.

---

## Run 1 post-only, run 2 pre+post

Post-only first to get the pipeline end-to-end. Then pre+post as the headline
experiment, because the no-damage/minor boundary is fundamentally change
detection and the human annotators had both images. Two runs give a real
ablation with a number attached rather than one guess.

---

## Metrics: accuracy is reported twice, on purpose

`accuracy_parsed` is skill given usable output. `accuracy_all` charges the model
for failures. The second is the deployment-relevant one, and reporting only the
first is the standard way to accidentally flatter a model with a high
unparseable rate.

QWK is the ordinal-agreement metric a remote-sensing reader will look for. MAE
distinguishes minor↔major confusion from no-damage↔destroyed. Adjacent and
distant error rates are reported separately for the same reason.

---

## Reproducibility: the template fingerprint

`prompts.template_fingerprint()` hashes the prompt version, system prompt, schema
description, both prompt shapes, the mask rate, and every evidence clause. It is
written into the dataset card and into each adapter's `run_config.json`, and
`predict.py` **refuses to run** an adapter whose fingerprint does not match the
current `prompts.py`.

This is the silent-failure the brief warns about: inference reconstructs the
prompt slightly differently, accuracy drops several points, and nothing in the
logs explains it.

---

## Open questions

- Does freezing the vision encoder cost more than expected on nadir imagery?
- Does `--outline` beat centring by enough to matter, and is that a shortcut?
- Does the minor/major boundary track published human annotator disagreement?
- Does per-event-type accuracy degrade on types with one training event?
