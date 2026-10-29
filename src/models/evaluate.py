"""Model evaluation routines and metrics computation for AttritionIQ.

Calculates ROC-AUC, PR-AUC, confusion matrices, calibration curves,
and group-split diagnostic metrics.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, Any]:
    """Calculates comprehensive classification performance metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))
    brier = float(brier_score_loss(y_true, y_proba))

    cm = confusion_matrix(y_true, y_pred).tolist()

    # ROC curve points (sampled for serialization)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    step_roc = max(1, len(fpr) // 50)
    roc_data = {
        "fpr": [float(x) for x in fpr[::step_roc]],
        "tpr": [float(x) for x in tpr[::step_roc]]
    }

    # PR curve points
    pr_p, pr_r, _ = precision_recall_curve(y_true, y_proba)
    step_pr = max(1, len(pr_p) // 50)
    pr_data = {
        "precision": [float(x) for x in pr_p[::step_pr]],
        "recall": [float(x) for x in pr_r[::step_pr]]
    }

    # Calibration curve
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        cal_data = {
            "prob_true": [float(x) for x in prob_true],
            "prob_pred": [float(x) for x in prob_pred]
        }
    except Exception:
        cal_data = {"prob_true": [], "prob_pred": []}

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "roc_curve": roc_data,
        "pr_curve": pr_data,
        "calibration_curve": cal_data
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Tuple[float, float]:
    """Finds decision threshold maximizing F1 score on validation set."""
    best_thresh = 0.50
    best_f1 = 0.0
    for t in np.linspace(0.1, 0.9, 81):
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thresh = float(round(t, 2))
    return best_thresh, round(best_f1, 4)
