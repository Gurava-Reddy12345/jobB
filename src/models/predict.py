"""Inference engine and risk categorization for AttritionIQ.

Handles single employee profile scoring, batch CSV predictions,
risk tier categorization, and actionable HR intervention recommendations.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import joblib
import pandas as pd
import numpy as np

from src.data.validation import validate_single_record, validate_employee_data
from src.features.preprocessing import engineer_features

DEFAULT_PIPELINE_PATH = os.path.join("artifacts", "model_pipeline.joblib")


def load_pipeline(pipeline_path: str = DEFAULT_PIPELINE_PATH) -> Dict[str, Any]:
    """Loads the serialized model pipeline artifact."""
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"Model pipeline not found at '{pipeline_path}'. "
            "Please train the model first by running: python src/models/train.py"
        )
    return joblib.load(pipeline_path)


def determine_risk_category(prob: float) -> str:
    """Maps predicted probability to actionable risk categories."""
    if prob < 0.35:
        return "Low Risk"
    elif prob < 0.65:
        return "Medium Risk"
    else:
        return "High Risk"


def generate_recommended_actions(record: Dict[str, Any], prob: float) -> List[str]:
    """Derives domain-informed HR review actions based on employee parameters."""
    actions = []
    hours = record.get("average_montly_hours", 200)
    projects = record.get("number_project", 3)
    satisfaction = record.get("satisfaction_level", 0.5)
    evaluation = record.get("last_evaluation", 0.5)
    salary = str(record.get("salary", "medium")).lower()
    tenure = record.get("time_spend_company", 3)
    promoted = record.get("promotion_last_5years", 0)

    if hours > 240 and projects >= 5:
        actions.append("Workload Rebalancing: Employee is working >240 hrs/month across heavy project load. High burnout probability.")
    elif hours < 140 and projects <= 2:
        actions.append("Engagement Review: Low monthly hours and project allocation indicate possible disengagement or underutilization.")

    if evaluation > 0.75 and satisfaction < 0.40:
        actions.append("Recognition & Compensation: High performer with low satisfaction. Review recognition, equity, and market salary parity.")

    if tenure in [3, 4, 5] and promoted == 0:
        actions.append("Career Pathway Discussion: Reached 3-5 year tenure milestone without promotion. Conduct career progression review.")

    if salary == "low" and evaluation >= 0.70:
        actions.append("Salary Benchmark: Strong evaluation score on 'low' compensation tier is a primary external poaching risk.")

    if not actions:
        if prob < 0.35:
            actions.append("Routine Check-in: Employee metrics indicate positive engagement. Maintain standard quarterly 1-on-1s.")
        else:
            actions.append("Holistic Review: Moderate attrition signal detected. Conduct manager touchpoint on team dynamics and career growth.")

    return actions


def predict_single(
    record: Dict[str, Any],
    pipeline: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Predicts attrition risk for an individual employee record.

    Returns:
        Structured prediction outcome including probability, risk category,
        recommendations, and audit metadata.
    """
    if pipeline is None:
        pipeline = load_pipeline()

    is_valid, errors, sanitized = validate_single_record(record)
    if not is_valid:
        return {
            "success": False,
            "errors": errors,
            "probability": None,
            "risk_category": None
        }

    # DataFrame conversion & feature engineering
    df_raw = pd.DataFrame([sanitized])
    df_featured = engineer_features(df_raw)

    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]
    decision_thresh = threshold if threshold is not None else pipeline.get("decision_threshold", 0.50)

    # Preprocess
    X_transformed = preprocessor.transform(df_featured)

    # Predict
    proba = float(model.predict_proba(X_transformed)[0, 1])
    is_attrition = bool(proba >= decision_thresh)
    risk_category = determine_risk_category(proba)
    recommendations = generate_recommended_actions(sanitized, proba)

    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "model_name": pipeline.get("model_name", "Random Forest"),
        "decision_threshold": decision_thresh,
        "estimated_probability": round(proba, 4),
        "risk_category": risk_category,
        "predicted_left": 1 if is_attrition else 0
    }

    return {
        "success": True,
        "errors": [],
        "probability": round(proba, 4),
        "probability_pct": round(proba * 100.0, 1),
        "is_attrition": is_attrition,
        "risk_category": risk_category,
        "decision_threshold": decision_thresh,
        "sanitized_record": sanitized,
        "featured_record": df_featured.iloc[0].to_dict(),
        "recommendations": recommendations,
        "audit": audit_entry
    }


def predict_batch(
    df: pd.DataFrame,
    pipeline: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Scores a batch DataFrame of employee records.

    Returns:
        Dictionary with valid predictions DataFrame, error logs, and summary statistics.
    """
    if pipeline is None:
        pipeline = load_pipeline()

    val_res = validate_employee_data(df, is_training=False)
    if not val_res["is_valid"]:
        return {
            "success": False,
            "errors": val_res["errors"],
            "warnings": val_res["warnings"],
            "predictions_df": pd.DataFrame(),
            "invalid_df": val_res["invalid_df"],
            "invalid_reasons": val_res["invalid_reasons"]
        }

    valid_df = val_res["valid_df"]
    df_featured = engineer_features(valid_df)

    preprocessor = pipeline["preprocessor"]
    model = pipeline["model"]
    decision_thresh = threshold if threshold is not None else pipeline.get("decision_threshold", 0.50)

    X_transformed = preprocessor.transform(df_featured)
    probas = model.predict_proba(X_transformed)[:, 1]

    output_df = valid_df.copy()
    output_df["estimated_risk_prob"] = np.round(probas, 4)
    output_df["estimated_risk_pct"] = np.round(probas * 100.0, 1)
    output_df["predicted_left"] = (probas >= decision_thresh).astype(int)
    output_df["risk_category"] = [determine_risk_category(p) for p in probas]

    # Quick primary driver heuristic for batch table
    drivers = []
    for _, row in df_featured.iterrows():
        if row.get("burnout_risk", 0) == 1:
            drivers.append("Burnout (High hours + Low satisfaction)")
        elif row.get("eval_sat_diff", 0) > 0.35:
            drivers.append("Performance/Satisfaction Disparity")
        elif row.get("average_montly_hours", 0) > 250:
            drivers.append("Excessive Overtime")
        elif row.get("time_spend_company", 0) in [3, 4, 5] and row.get("salary", "") == "low":
            drivers.append("Tenure Plateau on Low Salary")
        elif row.get("satisfaction_level", 0.5) < 0.35:
            drivers.append("Low Job Satisfaction")
        else:
            drivers.append("Standard Baseline")
    output_df["primary_risk_driver"] = drivers

    return {
        "success": True,
        "errors": val_res["errors"],
        "warnings": val_res["warnings"],
        "predictions_df": output_df,
        "invalid_df": val_res["invalid_df"],
        "invalid_reasons": val_res["invalid_reasons"],
        "total_records": len(df),
        "valid_count": len(valid_df),
        "invalid_count": len(val_res["invalid_df"]),
        "high_risk_count": int((output_df["risk_category"] == "High Risk").sum()),
        "medium_risk_count": int((output_df["risk_category"] == "Medium Risk").sum()),
        "low_risk_count": int((output_df["risk_category"] == "Low Risk").sum()),
        "avg_risk_probability": round(float(np.mean(probas)), 4) if len(probas) > 0 else 0.0
    }
