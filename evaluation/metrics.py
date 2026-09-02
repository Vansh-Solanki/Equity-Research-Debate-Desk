"""Pipeline-quality metrics: macro precision/recall/F1 for the claim-entailment
pipeline against a hand-labeled test set (Phase 6's acceptance criterion is
F1 >= 0.75), plus a groundedness % helper that any list of checked claims reduces
to. Groundedness is used more fully by Phase 8's evaluation dashboard, but it
falls straight out of this pipeline's output, so it lives here rather than being
duplicated later.
"""

LABELS = ("supported", "contradicted", "not_enough_evidence")


def groundedness_pct(claims: list[dict]) -> float:
    """Fraction of `claims` (as returned by evaluation.pipeline.check_statement)
    labeled "supported". Returns 0.0 for an empty list rather than dividing by zero."""
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if c["entailment_label"] == "supported")
    return supported / len(claims)


def score_predictions(predicted: list[str], expected: list[str]) -> dict:
    """Macro-averaged precision/recall/F1 across LABELS, plus accuracy and a
    per-label breakdown. `predicted` and `expected` must be same-length, same-order
    label lists (one entry per test-set claim).
    """
    if len(predicted) != len(expected):
        raise ValueError("predicted and expected must be the same length")
    if not predicted:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "per_label": {}}

    per_label = {}
    for label in LABELS:
        tp = sum(1 for p, e in zip(predicted, expected) if p == label and e == label)
        fp = sum(1 for p, e in zip(predicted, expected) if p == label and e != label)
        fn = sum(1 for p, e in zip(predicted, expected) if p != label and e == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    accuracy = sum(1 for p, e in zip(predicted, expected) if p == e) / len(predicted)
    macro_precision = sum(m["precision"] for m in per_label.values()) / len(LABELS)
    macro_recall = sum(m["recall"] for m in per_label.values()) / len(LABELS)
    macro_f1 = sum(m["f1"] for m in per_label.values()) / len(LABELS)

    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "per_label": per_label,
    }
