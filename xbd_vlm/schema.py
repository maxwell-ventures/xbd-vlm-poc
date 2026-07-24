"""The output schema, and the only code allowed to read or write it.

Every training example, every model generation, and every evaluation pass goes
through this module. If the schema changes, it changes in exactly one place and
`SCHEMA_VERSION` goes up.

Wire format — three fields, fixed order, one per line:

    DAMAGE: major-damage
    EVIDENCE: Partial roof failure with visible structural deformation; ...
    PRIORITY: high

Chosen over JSON deliberately: fewer tokens spent on punctuation, and a
malformed generation degrades into "one field missing" rather than "the whole
object failed to parse", which makes the unparseable-rate metric more
informative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SCHEMA_VERSION = "v1"

# Ordinal, low to high. These are xBD's own subtype strings, used verbatim so
# there is no translation layer between the labels and the model's vocabulary.
DAMAGE_GRADES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
GRADE_TO_ORDINAL = {g: i for i, g in enumerate(DAMAGE_GRADES)}

# xBD also emits "un-classified" for buildings the annotators could not grade.
# Those rows are dropped at parse time rather than mapped to a grade.
UNGRADED_SUBTYPE = "un-classified"

PRIORITIES = ["none", "moderate", "high", "critical"]


def priority_for(grade: str, days_since: int | None) -> str:
    """Triage priority as a stated rule, not a learned judgement.

    This is a deterministic function of damage grade and time since the event.
    It is a policy we are imposing, and the writeup must say so — the model is
    not inferring urgency, it is applying a rubric we taught it.

    The reason it is worth including at all: `days_since` only reaches the model
    through the prompt text. So the priority field is the one place where the
    model provably has to *read the context* to be correct, which gives the
    context-conditioning demo something measurable rather than anecdotal.
    """
    if grade == "no-damage":
        return "none"
    if grade == "minor-damage":
        return "moderate"
    if grade == "major-damage":
        # Trapped-occupant risk is concentrated in the first week.
        return "critical" if (days_since is not None and days_since <= 7) else "high"
    if grade == "destroyed":
        return "critical" if (days_since is not None and days_since <= 14) else "high"
    raise ValueError(f"unknown damage grade: {grade!r}")


@dataclass
class Assessment:
    damage: str
    evidence: str
    priority: str

    def to_text(self) -> str:
        return (
            f"DAMAGE: {self.damage}\n"
            f"EVIDENCE: {self.evidence}\n"
            f"PRIORITY: {self.priority}"
        )


@dataclass
class ParseResult:
    """Outcome of reading a model generation.

    `assessment` is None when the output could not be read. `reason` then holds
    a short machine-comparable tag so failures can be counted by kind rather
    than lumped into a single unparseable number.
    """

    assessment: Assessment | None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.assessment is not None


def _field_re(name: str) -> re.Pattern:
    """Match `NAME: value` tolerantly.

    Instruct-tuned models decorate output even when told not to: leading bullets,
    `**bold**` keys, heading markers. None of that is a schema violation worth
    counting, so it is stripped here rather than charged to the unparseable rate.
    Anything beyond this — prose, missing fields, invented vocabulary — still
    fails, which is the behaviour the metric is meant to capture.
    """
    return re.compile(
        rf"^[\s*#>\-•]*{name}\s*\**\s*:\s*\**\s*(.+?)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )


_FIELD_RE = {
    "damage": _field_re("DAMAGE"),
    "evidence": _field_re("EVIDENCE"),
    "priority": _field_re("PRIORITY"),
}

# Zero-shot models tend to answer in prose or with near-miss vocabulary. We
# normalise a small, explicitly enumerated set of aliases so the baseline is not
# penalised for cosmetic differences — anything beyond this list counts as
# unparseable, and that is the honest result.
_GRADE_ALIASES = {
    "no damage": "no-damage",
    "none": "no-damage",
    "undamaged": "no-damage",
    "no_damage": "no-damage",
    "minor damage": "minor-damage",
    "minor": "minor-damage",
    "minor_damage": "minor-damage",
    "major damage": "major-damage",
    "major": "major-damage",
    "major_damage": "major-damage",
    "destroyed": "destroyed",
    "total destruction": "destroyed",
}

_PRIORITY_ALIASES = {
    "no priority": "none",
    "low": "moderate",
    "medium": "moderate",
    "moderate": "moderate",
    "high": "high",
    "critical": "critical",
    "urgent": "critical",
    "immediate": "critical",
}


def _normalise(value: str, aliases: dict[str, str], valid: list[str]) -> str | None:
    v = value.strip().strip(" .*_`\"'").strip().lower()
    if v in valid:
        return v
    return aliases.get(v)


def parse_response(text: str) -> ParseResult:
    """Read a model generation into an Assessment, or explain why not."""
    if not text or not text.strip():
        return ParseResult(None, "empty")

    raw = {}
    for field, pattern in _FIELD_RE.items():
        m = pattern.search(text)
        if m is None:
            return ParseResult(None, f"missing:{field}")
        raw[field] = m.group(1)

    damage = _normalise(raw["damage"], _GRADE_ALIASES, DAMAGE_GRADES)
    if damage is None:
        return ParseResult(None, "bad-value:damage")

    priority = _normalise(raw["priority"], _PRIORITY_ALIASES, PRIORITIES)
    if priority is None:
        return ParseResult(None, "bad-value:priority")

    evidence = raw["evidence"].strip()
    if not evidence:
        return ParseResult(None, "empty:evidence")

    return ParseResult(Assessment(damage=damage, evidence=evidence, priority=priority))


def schema_description() -> str:
    """The schema, spelled out for the model.

    This string is part of the prompt for BOTH the tuned model and the zero-shot
    baseline. Withholding it from the baseline would inflate the delta with
    format compliance rather than assessment quality.
    """
    grades = " | ".join(DAMAGE_GRADES)
    prios = " | ".join(PRIORITIES)
    return (
        "Respond with exactly three lines, in this order, and nothing else:\n"
        f"DAMAGE: <one of: {grades}>\n"
        "EVIDENCE: <one sentence naming the visual features that determined the grade>\n"
        f"PRIORITY: <one of: {prios}>"
    )
