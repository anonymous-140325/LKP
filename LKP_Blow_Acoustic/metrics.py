"""
metrics.py

Biometric evaluation metrics.

FAR and FRR are computed as proportions of the total number of
comparisons rather than per-class rates.

    FAR = FP / (TP + TN + FP + FN)
    FRR = FN / (TP + TN + FP + FN)
    EER = (FAR + FRR) / 2
"""


def compute_accuracy(tp, tn, fp, fn):
    return (tp + tn) / max(tp + tn + fp + fn, 1)


def compute_far(tp, tn, fp, fn):
    return fp / max(tp + tn + fp + fn, 1)


def compute_frr(tp, tn, fp, fn):
    return fn / max(tp + tn + fp + fn, 1)


def compute_eer(far, frr):
    return (far + frr) / 2


def compute_metrics(tp, tn, fp, fn):
    """
    Convenience wrapper that returns all metrics in a single dict.
    """

    accuracy = compute_accuracy(tp, tn, fp, fn)
    far      = compute_far(tp, tn, fp, fn)
    frr      = compute_frr(tp, tn, fp, fn)
    eer      = compute_eer(far, frr)

    return {
        "Accuracy": accuracy,
        "FAR":      far,
        "FRR":      frr,
        "EER":      eer
    }
