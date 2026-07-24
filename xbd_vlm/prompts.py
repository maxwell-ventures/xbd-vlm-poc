"""Prompt construction and target-text templating.

Two responsibilities, kept together because they must stay in lockstep:

1. `build_prompt` — the exact text the model sees, at training and inference.
   An adapter is meaningless without the template it was trained on, so this is
   versioned and hashed into every run config.

2. `build_evidence` — the EVIDENCE field of the training target.

## About the evidence field

xBD supplies a polygon and an ordinal grade. It supplies no rationale text. So
the evidence sentences here are TEMPLATED: composed from a damage-grade clause
and an event-type clause. They are not human annotations and they are not
grounded in the specific pixels of any specific chip.

What that means, stated plainly so the writeup can repeat it: the evidence field
teaches output format and teaches the model to associate event context with
damage vocabulary. It does not teach grounded visual reasoning, and a correct
evidence sentence is not evidence that the model looked at the right thing.

The honest upgrade path is distillation — have a stronger VLM write the
rationale for each chip conditioned on the known grade — which is deliberately
out of scope here.

## Why the templates are event-conditioned

If EVIDENCE depended only on damage grade, the event-type field in the prompt
would carry no information about the target, and the model would be free to
ignore it. Conditioning the phrasing on event type means the prompt field is
load-bearing. Combined with `days_since` driving PRIORITY (see schema.py), both
context fields have a measurable effect on the correct answer.

This is by construction, not emergent. The demo shows that the model learned the
conditioning we built in; it does not show that the model discovered it.
"""

from __future__ import annotations

import hashlib

from .schema import DAMAGE_GRADES, schema_description

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You are a damage assessment analyst reviewing post-disaster satellite "
    "imagery. You grade individual structures on a four-level scale and report "
    "the visual evidence behind each grade."
)

# Fraction of training examples where event type is withheld. Without this the
# model can learn to read event type off the pixels and ignore the prompt; the
# withheld examples make the field genuinely informative. See notes/decisions.md
UNKNOWN_EVENT_RATE = 0.15


def build_prompt(
    event_type: str,
    days_since: int | None,
    structure_type: str | None = None,
    has_pre_image: bool = False,
) -> str:
    """The user-turn text. Identical at train and inference time."""
    if has_pre_image:
        framing = (
            "The first image is the same location before the event; the second "
            "is after. In both, the structure to assess is at the centre."
        )
    else:
        framing = "The structure to assess is at the centre of the image."

    lines = [
        "Assess the structure at the centre of this post-event satellite chip.",
        "",
        framing,
        "",
        "Context:",
        f"- Event type: {event_type}",
    ]
    if days_since is not None:
        lines.append(f"- Days since event: {days_since}")
    else:
        lines.append("- Days since event: unknown")
    lines.append(f"- Structure type: {structure_type or 'unknown'}")
    lines += ["", schema_description()]
    return "\n".join(lines)


# --- Evidence templating -------------------------------------------------

# Structural observation, keyed by damage grade. Three variants each so the
# training targets are not a single memorised string per class.
_GRADE_CLAUSES: dict[str, list[str]] = {
    "no-damage": [
        "Roofline intact and continuous, with no displaced material",
        "Structure outline unbroken and roof surface uniform",
        "No visible breach of the roof plane or structural envelope",
    ],
    "minor-damage": [
        "Localised roof surface disruption with the structure otherwise intact",
        "Partial loss of roof covering over a limited area, walls sound",
        "Scattered surface damage to the roof, overall footprint unchanged",
    ],
    "major-damage": [
        "Substantial roof failure exposing the structure's interior",
        "Partial collapse with visible deformation of the building footprint",
        "Large sections of roof missing and structural members displaced",
    ],
    "destroyed": [
        "Structure reduced to rubble with the original footprint no longer legible",
        "Complete collapse, only debris and foundation remaining",
        "No standing structure present within the original footprint",
    ],
}

# Environmental cue, keyed by event type. Applied to every grade — a
# "no-damage" building surrounded by floodwater is a real and important case.
_EVENT_CLAUSES: dict[str, list[str]] = {
    "hurricane": [
        "wind-scattered debris across the surrounding parcels",
        "debris fields aligned along the prevailing wind direction",
        "damage to adjacent tree cover and outbuildings",
    ],
    "tornado": [
        "a narrow corridor of destruction crossing neighbouring lots",
        "abrupt transition to undamaged structures a short distance away",
        "debris scatter concentrated along a single track",
    ],
    "wildfire": [
        "burn scars and ash across the surrounding vegetation",
        "charring on adjacent parcels with vegetation reduced to bare ground",
        "a sharp burn perimeter running through the neighbouring properties",
    ],
    "flood": [
        "standing water surrounding the structure",
        "inundation across the adjacent parcels and roadways",
        "sediment deposition and waterlines visible in the surrounding area",
    ],
    "tsunami": [
        "widespread scouring and debris rafted inland from the shoreline",
        "sediment wash and displaced material across the surrounding area",
        "clearance of lighter structures in the surrounding block",
    ],
    "earthquake": [
        "adjacent structures showing comparable collapse patterns",
        "rubble spilling into the surrounding street network",
        "no environmental disturbance beyond the structures themselves",
    ],
    "volcanic-eruption": [
        "ash mantling the surrounding terrain",
        "lava or lahar deposits encroaching on the adjacent parcels",
        "burial of surrounding ground surface under deposited material",
    ],
}


def _pick(options: list[str], key: str, salt: str) -> str:
    """Deterministic choice, stable across regenerations of the dataset."""
    h = hashlib.sha1(f"{key}|{salt}".encode()).digest()
    return options[h[0] % len(options)]


def build_evidence(grade: str, event_type: str, uid: str) -> str:
    """Compose the templated EVIDENCE sentence for a training target."""
    if grade not in _GRADE_CLAUSES:
        raise ValueError(f"unknown damage grade: {grade!r}")
    structural = _pick(_GRADE_CLAUSES[grade], uid, "structural")
    cues = _EVENT_CLAUSES.get(event_type)
    if not cues:
        return f"{structural}."
    return f"{structural}; {_pick(cues, uid, 'event')}."


def template_fingerprint() -> str:
    """Hash covering everything that shapes prompt and target text.

    Written into every run config. If this changes between training and
    inference, the adapter is being used with a template it never saw.
    """
    parts = [
        PROMPT_VERSION,
        SYSTEM_PROMPT,
        schema_description(),
        build_prompt("hurricane", 7, "residential", has_pre_image=False),
        build_prompt("hurricane", 7, "residential", has_pre_image=True),
        str(UNKNOWN_EVENT_RATE),
    ]
    for g in DAMAGE_GRADES:
        parts.extend(_GRADE_CLAUSES[g])
    for e in sorted(_EVENT_CLAUSES):
        parts.extend(_EVENT_CLAUSES[e])
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:16]
