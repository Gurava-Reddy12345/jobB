"""Unit tests for feature engineering and preprocessor pipeline."""

import pytest
import pandas as pd
from src.features.preprocessing import (
    engineer_features,
    clean_pct_series,
    build_preprocessor,
    ENGINEERED_FEATURES
)


def test_clean_pct_series():
    s = pd.Series(["45%", "85%", 0.35, "100%", 75])
    cleaned = clean_pct_series(s)
    assert cleaned.iloc[0] == pytest.approx(0.45)
    assert cleaned.iloc[1] == pytest.approx(0.85)
    assert cleaned.iloc[2] == pytest.approx(0.35)
    assert cleaned.iloc[3] == pytest.approx(1.0)
    assert cleaned.iloc[4] == pytest.approx(0.75)


def test_engineer_features():
    df = pd.DataFrame([
        {
            "satisfaction_level": 0.20,
            "last_evaluation": 0.90,
            "number_project": 6,
            "average_montly_hours": 260,
            "time_spend_company": 4,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "Department": "technical",
            "salary": "low"
        }
    ])
    feat = engineer_features(df)

    # Check that all engineered parameters are present
    for col in ENGINEERED_FEATURES:
        assert col in feat.columns

    # Check calculations
    assert feat["hours_per_project"].iloc[0] == pytest.approx(260 / 6)
    assert feat["eval_sat_diff"].iloc[0] == pytest.approx(0.90 - 0.20)
    assert feat["overworked"].iloc[0] == 1
    assert feat["burnout_risk"].iloc[0] == 1
    assert feat["unrewarded_high_performer"].iloc[0] == 1
    assert feat["tenure_risk_window"].iloc[0] == 1
    assert feat["salary_num"].iloc[0] == 1


def test_preprocessor_pipeline():
    df = pd.DataFrame([
        {
            "satisfaction_level": 0.35,
            "last_evaluation": 0.65,
            "number_project": 3,
            "average_montly_hours": 180,
            "time_spend_company": 2,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "Department": "sales",
            "salary": "medium"
        },
        {
            "satisfaction_level": 0.85,
            "last_evaluation": 0.95,
            "number_project": 5,
            "average_montly_hours": 240,
            "time_spend_company": 5,
            "Work_accident": 1,
            "promotion_last_5years": 1,
            "Department": "management",
            "salary": "high"
        }
    ])
    featured = engineer_features(df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(featured)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 10
