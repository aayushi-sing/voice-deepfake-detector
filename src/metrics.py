"""
metrics.py
Shared evaluation metrics, used by both train.py (for checkpointing/early
stopping) and eval.py (for the final reported numbers). Keeping this in one
place means the EER computed during training and the EER reported in your
final results are guaranteed to be computed the same way.
"""

import numpy as np
from sklearn.metrics import (
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_eer(y_true, y_scores):
    """
    Compute the Equal Error Rate (EER) -- the standard metric for spoof/
    anti-spoofing detection (used throughout the ASVspoof literature).

    EER is the point on the ROC curve where the False Positive Rate (FPR)
    equals the False Negative Rate (FNR). Lower is better. Unlike Accuracy/
    F1, EER doesn't depend on picking a decision threshold -- it summarizes
    performance across all thresholds, which is why the field reports it.

    Args:
        y_true:   array-like of 0/1 ground-truth labels (1 = fake/spoof)
        y_scores: array-like of predicted probabilities/scores for class 1

    Returns:
        eer: float in [0, 1] (multiply by 100 for a percentage)
        threshold: the score threshold at which FPR == FNR
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr

    # Find the threshold where FPR and FNR are closest (their crossing point)
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)  # average of the two at the crossing
    eer_threshold = float(thresholds[idx])

    return eer, eer_threshold


def compute_classification_metrics(y_true, y_scores, threshold=0.5):
    """
    Compute Accuracy, Precision, Recall, F1 at a fixed decision threshold
    (default 0.5), plus a confusion matrix. This is the standard ML-eval
    complement to EER.

    Args:
        y_true:   array-like of 0/1 ground-truth labels
        y_scores: array-like of predicted probabilities for class 1 (fake)
        threshold: decision threshold applied to y_scores

    Returns:
        dict with keys: accuracy, precision, recall, f1, confusion_matrix
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def full_report(y_true, y_scores, threshold=0.5):
    """
    Convenience wrapper: returns both classification metrics and EER in one
    dict, and a pre-formatted string ready to print or write to a file.
    """
    cls_metrics = compute_classification_metrics(y_true, y_scores, threshold)
    eer, eer_threshold = compute_eer(y_true, y_scores)

    report = dict(cls_metrics)
    report["eer"] = eer
    report["eer_threshold"] = eer_threshold

    lines = [
        "=== Evaluation Report ===",
        f"Accuracy:  {cls_metrics['accuracy']:.4f}",
        f"Precision: {cls_metrics['precision']:.4f}",
        f"Recall:    {cls_metrics['recall']:.4f}",
        f"F1-score:  {cls_metrics['f1']:.4f}",
        f"EER:       {eer * 100:.2f}%  (threshold={eer_threshold:.4f})",
        f"Confusion Matrix (rows=true, cols=pred, order=[real, fake]):",
        f"  {cls_metrics['confusion_matrix']}",
    ]
    report["text"] = "\n".join(lines)
    return report


# ---------------------------------------------------------------------------
# Smoke test -- pure numpy/sklearn, no torch/librosa needed, runs anywhere.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic example: a decent-but-imperfect classifier
    rng = np.random.default_rng(42)
    n = 500
    y_true = rng.integers(0, 2, size=n)
    # scores correlated with truth but noisy, to simulate a real classifier
    y_scores = np.clip(y_true * 0.6 + rng.normal(0, 0.3, size=n) + 0.2, 0, 1)

    report = full_report(y_true, y_scores)
    print(report["text"])

    assert 0 <= report["eer"] <= 1
    assert 0 <= report["accuracy"] <= 1
    print("\nSmoke test passed.")
