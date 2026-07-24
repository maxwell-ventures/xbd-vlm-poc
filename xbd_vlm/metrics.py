"""Metrics for an ordinal four-level damage scale.

No sklearn dependency — these are short enough to write out, and writing them
out means knowing exactly what is being counted, which matters when the headline
number is a delta between two runs.

Unparseable generations are counted, never dropped. A model that emits prose on
20% of inputs is worse than one that does not, and silently excluding those rows
would hide it.
"""

from __future__ import annotations

from collections import Counter

from .schema import DAMAGE_GRADES, GRADE_TO_ORDINAL, PRIORITIES

N = len(DAMAGE_GRADES)


def confusion_matrix(pairs: list[tuple[str, str]]) -> list[list[int]]:
    """rows = true grade, cols = predicted grade."""
    m = [[0] * N for _ in range(N)]
    for truth, pred in pairs:
        m[GRADE_TO_ORDINAL[truth]][GRADE_TO_ORDINAL[pred]] += 1
    return m


def per_class(pairs: list[tuple[str, str]]) -> dict[str, dict[str, float]]:
    m = confusion_matrix(pairs)
    out = {}
    for i, grade in enumerate(DAMAGE_GRADES):
        tp = m[i][i]
        support = sum(m[i])
        predicted = sum(m[r][i] for r in range(N))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[grade] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return out


def quadratic_weighted_kappa(pairs: list[tuple[str, str]]) -> float:
    """Agreement corrected for chance, penalising distant confusions quadratically.

    The standard metric for ordinal agreement, and the one a remote-sensing
    reader will look for. 0 = chance, 1 = perfect, negative = worse than chance.
    """
    if not pairs:
        return 0.0
    observed = confusion_matrix(pairs)
    total = len(pairs)

    true_counts = Counter(GRADE_TO_ORDINAL[t] for t, _ in pairs)
    pred_counts = Counter(GRADE_TO_ORDINAL[p] for _, p in pairs)

    num = 0.0
    den = 0.0
    for i in range(N):
        for j in range(N):
            w = ((i - j) ** 2) / ((N - 1) ** 2)
            expected = true_counts[i] * pred_counts[j] / total
            num += w * observed[i][j]
            den += w * expected
    if den == 0:
        return 0.0
    return 1.0 - num / den


def score(
    records: list[dict],
) -> dict:
    """Compute the full metric block.

    Each record needs: `true_damage`, `pred_damage` (None if unparseable),
    and optionally `true_priority` / `pred_priority` and `parse_reason`.
    """
    total = len(records)
    parsed = [r for r in records if r.get("pred_damage")]
    unparseable = [r for r in records if not r.get("pred_damage")]

    pairs = [(r["true_damage"], r["pred_damage"]) for r in parsed]

    exact = sum(1 for t, p in pairs if t == p)
    dist = [abs(GRADE_TO_ORDINAL[t] - GRADE_TO_ORDINAL[p]) for t, p in pairs]
    adjacent = sum(1 for d in dist if d == 1)
    distant = sum(1 for d in dist if d >= 2)

    pc = per_class(pairs)
    macro_f1 = sum(v["f1"] for v in pc.values()) / N

    # Two accuracy denominators, both reported. `accuracy_parsed` is the model's
    # skill given it produced usable output; `accuracy_all` charges it for the
    # failures. The second is the deployment-relevant one.
    result = {
        "n_total": total,
        "n_parsed": len(parsed),
        "unparseable_rate": len(unparseable) / total if total else 0.0,
        "parse_failure_reasons": dict(
            Counter(r.get("parse_reason", "unknown") for r in unparseable)
        ),
        "accuracy_parsed": exact / len(parsed) if parsed else 0.0,
        "accuracy_all": exact / total if total else 0.0,
        "adjacent_error_rate": adjacent / len(parsed) if parsed else 0.0,
        "distant_error_rate": distant / len(parsed) if parsed else 0.0,
        "mae_ordinal": sum(dist) / len(dist) if dist else 0.0,
        "macro_f1": macro_f1,
        "qwk": quadratic_weighted_kappa(pairs),
        "per_class": pc,
        "confusion_matrix": confusion_matrix(pairs),
        "confusion_labels": DAMAGE_GRADES,
    }

    prio = [
        r for r in parsed if r.get("true_priority") and r.get("pred_priority")
    ]
    if prio:
        result["priority_accuracy"] = sum(
            1 for r in prio if r["true_priority"] == r["pred_priority"]
        ) / len(prio)
        result["priority_labels"] = PRIORITIES
    return result


def format_report(name: str, m: dict) -> str:
    """Human-readable metric block for the terminal and the writeup."""
    lines = [
        f"=== {name} ===",
        f"  examples            {m['n_total']}",
        f"  unparseable         {m['unparseable_rate']:.1%}  ({m['n_total'] - m['n_parsed']} of {m['n_total']})",
    ]
    if m["parse_failure_reasons"]:
        for reason, count in sorted(
            m["parse_failure_reasons"].items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"      {reason:<20} {count}")
    lines += [
        f"  accuracy (parsed)   {m['accuracy_parsed']:.3f}",
        f"  accuracy (all)      {m['accuracy_all']:.3f}",
        f"  macro F1            {m['macro_f1']:.3f}",
        f"  QWK                 {m['qwk']:.3f}",
        f"  ordinal MAE         {m['mae_ordinal']:.3f}",
        f"  adjacent errors     {m['adjacent_error_rate']:.1%}",
        f"  distant errors      {m['distant_error_rate']:.1%}",
    ]
    if "priority_accuracy" in m:
        lines.append(f"  priority accuracy   {m['priority_accuracy']:.3f}")

    lines.append("")
    lines.append(f"  {'class':<16}{'prec':>7}{'rec':>7}{'f1':>7}{'n':>8}")
    for grade, v in m["per_class"].items():
        lines.append(
            f"  {grade:<16}{v['precision']:>7.3f}{v['recall']:>7.3f}"
            f"{v['f1']:>7.3f}{v['support']:>8}"
        )

    lines.append("")
    lines.append("  confusion (rows=true, cols=pred)")
    header = "".join(f"{g[:9]:>11}" for g in m["confusion_labels"])
    lines.append(f"  {'':<16}{header}")
    for grade, row in zip(m["confusion_labels"], m["confusion_matrix"]):
        cells = "".join(f"{c:>11}" for c in row)
        lines.append(f"  {grade:<16}{cells}")
    return "\n".join(lines)


def format_delta(base: dict, tuned: dict) -> str:
    """The headline table: base vs tuned, per class."""
    lines = [
        "=== base vs tuned ===",
        f"  {'metric':<22}{'base':>10}{'tuned':>10}{'delta':>10}",
    ]
    for key, label, fmt in [
        ("accuracy_all", "accuracy (all)", "{:.3f}"),
        ("macro_f1", "macro F1", "{:.3f}"),
        ("qwk", "QWK", "{:.3f}"),
        ("mae_ordinal", "ordinal MAE (lower)", "{:.3f}"),
        ("unparseable_rate", "unparseable", "{:.3f}"),
    ]:
        b, t = base.get(key, 0.0), tuned.get(key, 0.0)
        lines.append(
            f"  {label:<22}{fmt.format(b):>10}{fmt.format(t):>10}{t - b:>+10.3f}"
        )
    lines.append("")
    lines.append(f"  per-class recall {'':<6}{'base':>10}{'tuned':>10}{'delta':>10}")
    for grade in base.get("per_class", {}):
        b = base["per_class"][grade]["recall"]
        t = tuned.get("per_class", {}).get(grade, {}).get("recall", 0.0)
        lines.append(f"  {grade:<22}{b:>10.3f}{t:>10.3f}{t - b:>+10.3f}")
    return "\n".join(lines)
