"""Unit tests for data validation and schema verification."""

import pytest
import pandas as pd
from src.data.validation import (
    validate_single_record,
    validate_employee_data,
    clean_percentage_value,
    get_data_quality_summary
)


def test_clean_percentage_value():
    assert clean_percentage_value("38%") == pytest.approx(0.38)
    assert clean_percentage_value("100%") == pytest.approx(1.0)
    assert clean_percentage_value(0.75) == pytest.approx(0.75)
    assert clean_percentage_value(80) == pytest.approx(0.80)
    assert clean_percentage_value("invalid") is None
    assert clean_percentage_value(None) is None


def test_validate_single_record_valid():
    rec = {
        "satisfaction_level": "45%",
        "last_evaluation": "78%",
        "number_project": 4,
        "average_montly_hours": 210,
        "time_spend_company": 3,
        "Work_accident": 0,
        "promotion_last_5years": 0,
        "Department": "sales",
        "salary": "medium"
    }
    is_valid, errors, sanitized = validate_single_record(rec)
    assert is_valid is True
    assert len(errors) == 0
    assert sanitized["satisfaction_level"] == pytest.approx(0.45)
    assert sanitized["last_evaluation"] == pytest.approx(0.78)
    assert sanitized["Department"] == "sales"
    assert sanitized["salary"] == "medium"


def test_validate_single_record_invalid_ranges():
    rec = {
        "satisfaction_level": 1.5,  # Invalid
        "last_evaluation": 0.8,
        "number_project": 25,  # Out of range
        "average_montly_hours": 10,  # Below 40
        "time_spend_company": 0,  # Below 1
        "Work_accident": 5,  # Non-binary
        "promotion_last_5years": 0,
        "Department": "astronomy",  # Unknown department
        "salary": "astronomical"  # Unknown salary
    }
    is_valid, errors, _ = validate_single_record(rec)
    assert is_valid is False
    assert any("satisfaction_level" in e for e in errors)
    assert any("number_project" in e for e in errors)
    assert any("average_montly_hours" in e for e in errors)
    assert any("Department" in e for e in errors)
    assert any("salary" in e for e in errors)


def test_validate_employee_data_batch():
    df = pd.DataFrame([
        {
            "satisfaction_level": 0.38,
            "last_evaluation": 0.53,
            "number_project": 2,
            "average_montly_hours": 157,
            "time_spend_company": 3,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "Department": "sales",
            "salary": "low",
            "left": 1
        },
        {
            "satisfaction_level": 0.80,
            "last_evaluation": 0.86,
            "number_project": 5,
            "average_montly_hours": 262,
            "time_spend_company": 6,
            "Work_accident": 0,
            "promotion_last_5years": 0,
            "Department": "technical",
            "salary": "medium",
            "left": 0
        }
    ])
    # In inference mode, 'left' should trigger target isolation warning
    res_inf = validate_employee_data(df, is_training=False)
    assert res_inf["is_valid"] is True
    assert res_inf["valid_count"] == 2
    assert res_inf["target_stripped"] is True

    # In training mode, 'left' is retained as target
    res_train = validate_employee_data(df, is_training=True)
    assert res_train["is_valid"] is True
    assert "left" in res_train["valid_df"].columns


def test_data_quality_summary():
    df = pd.DataFrame({
        "Emp_Id": ["A1", "A2"],
        "satisfaction_level": [0.4, 0.8],
        "last_evaluation": [0.6, 0.9],
        "number_project": [3, 4],
        "average_montly_hours": [180, 220],
        "time_spend_company": [2, 3],
        "Work_accident": [0, 0],
        "promotion_last_5years": [0, 0],
        "Department": ["sales", "hr"],
        "salary": ["low", "high"],
        "left": [1, 0]
    })
    summary = get_data_quality_summary(df)
    assert summary["total_records"] == 2
    assert summary["total_missing_values"] == 0
    assert summary["target_stats"]["left_count"] == 1
    assert "Emp_Id" in summary["excluded_features"]
