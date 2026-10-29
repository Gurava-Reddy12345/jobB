"""Unit tests for prediction service and risk categorization."""

import pytest
import pandas as pd
from src.models.predict import (
    determine_risk_category,
    generate_recommended_actions
)
from src.models.evaluate import find_optimal_threshold, calculate_classification_metrics
import numpy as np


def test_determine_risk_category():
    assert determine_risk_category(0.15) == "Low Risk"
    assert determine_risk_category(0.349) == "Low Risk"
    assert determine_risk_category(0.35) == "Medium Risk"
    assert determine_risk_category(0.50) == "Medium Risk"
    assert determine_risk_category(0.649) == "Medium Risk"
    assert determine_risk_category(0.65) == "High Risk"
    assert determine_risk_category(0.92) == "High Risk"


def test_generate_recommended_actions_burnout():
    record = {
        "average_montly_hours": 270,
        "number_project": 6,
        "satisfaction_level": 0.25,
        "last_evaluation": 0.85,
        "salary": "low",
        "time_spend_company": 4,
        "promotion_last_5years": 0
    }
    actions = generate_recommended_actions(record, prob=0.88)
    assert len(actions) > 0
    # Overwork & Recognition actions should be triggered
    assert any("Workload Rebalancing" in a for a in actions)
    assert any("Recognition & Compensation" in a for a in actions)


def test_find_optimal_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.6])
    thresh, f1 = find_optimal_threshold(y_true, y_proba)
    assert 0.1 <= thresh <= 0.9
    assert f1 > 0.8


def test_calculate_classification_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert "confusion_matrix" in metrics
