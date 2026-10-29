"""Feature engineering and transformation pipeline for AttritionIQ.

Transforms raw employee metrics into high-signal predictive features
including burnout indicators, workload ratios, and tenure friction windows.
"""

from typing import List, Dict, Any, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_NUMERIC_FEATURES = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years"
]

RAW_CATEGORICAL_FEATURES = [
    "Department",
    "salary"
]

ENGINEERED_FEATURES = [
    "hours_per_project",
    "eval_sat_diff",
    "projects_per_year",
    "overworked",
    "underworked",
    "burnout_risk",
    "unrewarded_high_performer",
    "tenure_risk_window",
    "salary_num",
    "salary_workload_ratio"
]

ALL_FEATURE_NAMES = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES + RAW_CATEGORICAL_FEATURES


def clean_pct_series(series: pd.Series) -> pd.Series:
    """Cleans a pandas series with potential percentages into float 0.0-1.0."""
    def _clean(val):
        if pd.isna(val):
            return 0.5
        if isinstance(val, (int, float)):
            f_val = float(val)
            if 1.0 < f_val <= 100.0:
                return f_val / 100.0
            return f_val
        if isinstance(val, str):
            clean_str = val.replace("%", "").strip()
            try:
                num = float(clean_str)
                if 1.0 < num <= 100.0:
                    return num / 100.0
                return num
            except ValueError:
                return 0.5
        return 0.5

    return series.apply(_clean)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derives high-signal domain features from validated input data.

    Engineered Parameters:
    - hours_per_project: Average hours committed to each concurrent project.
    - eval_sat_diff: Gap between performance evaluation and employee satisfaction.
    - projects_per_year: Project velocity relative to tenure.
    - overworked: Flag for employees working > 240 hrs with >= 5 projects.
    - underworked: Flag for employees working < 150 hrs with <= 2 projects.
    - burnout_risk: High evaluation (>0.75) + high hours (>220) + low satisfaction (<0.45).
    - unrewarded_high_performer: Top evaluation (>0.80) with no promotion & non-high salary.
    - tenure_risk_window: Indicator for tenure in 3-5 years (peak attrition cohort).
    - salary_num: Ordinal mapping (low=1, medium=2, high=3).
    - salary_workload_ratio: Normalized salary level relative to hours worked.
    """
    data = df.copy()

    # Clean percentage strings
    if "satisfaction_level" in data.columns:
        data["satisfaction_level"] = clean_pct_series(data["satisfaction_level"])
    if "last_evaluation" in data.columns:
        data["last_evaluation"] = clean_pct_series(data["last_evaluation"])

    # Ensure numeric types
    for col in RAW_NUMERIC_FEATURES:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    # 1. Hours per project
    num_proj = data["number_project"].replace(0, 1)
    data["hours_per_project"] = data["average_montly_hours"] / num_proj

    # 2. Evaluation vs Satisfaction differential (Underappreciation gap)
    data["eval_sat_diff"] = data["last_evaluation"] - data["satisfaction_level"]

    # 3. Project velocity per year
    tenure = data["time_spend_company"].replace(0, 1)
    data["projects_per_year"] = data["number_project"] / tenure

    # 4. Overworked indicator
    data["overworked"] = (
        (data["average_montly_hours"] > 240) & (data["number_project"] >= 5)
    ).astype(int)

    # 5. Underworked / disengagement indicator
    data["underworked"] = (
        (data["average_montly_hours"] < 150) & (data["number_project"] <= 2)
    ).astype(int)

    # 6. Burnout risk
    data["burnout_risk"] = (
        (data["last_evaluation"] > 0.75)
        & (data["average_montly_hours"] > 220)
        & (data["satisfaction_level"] < 0.45)
    ).astype(int)

    # 7. Unrewarded high performer
    salary_str = data["salary"].astype(str).str.lower()
    data["unrewarded_high_performer"] = (
        (data["last_evaluation"] > 0.8)
        & (data["promotion_last_5years"] == 0)
        & (salary_str != "high")
    ).astype(int)

    # 8. Tenure risk window (historical peak attrition is 3-5 years)
    data["tenure_risk_window"] = (
        data["time_spend_company"].isin([3, 4, 5])
    ).astype(int)

    # 9. Ordinal salary mapping
    salary_map = {"low": 1, "medium": 2, "high": 3}
    data["salary_num"] = salary_str.map(salary_map).fillna(1).astype(int)

    # 10. Salary-to-workload ratio
    hours_norm = (data["average_montly_hours"] / 100.0).replace(0, 1.0)
    data["salary_workload_ratio"] = data["salary_num"] / hours_norm

    return data


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer that applies feature engineering."""

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)
        return engineer_features(df)


def build_preprocessor() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for one-hot encoding and scaling."""
    numeric_cols = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
    categorical_cols = RAW_CATEGORICAL_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_cols
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols
            )
        ],
        remainder="drop"
    )
    return preprocessor
