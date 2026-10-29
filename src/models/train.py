"""Model training, benchmark comparison, and artifact serialization for AttritionIQ.

Trains and compares:
1. Logistic Regression (Linear baseline)
2. Random Forest Classifier (Bagged tree ensemble)
3. HistGradientBoostingClassifier (Advanced Gradient Boosting)

Includes Group-based splitting diagnostics to verify reliability against
repeated feature profiles.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
import shap

from src.data.validation import validate_employee_data, get_data_quality_summary
from src.features.preprocessing import (
    engineer_features,
    build_preprocessor,
    RAW_NUMERIC_FEATURES,
    ENGINEERED_FEATURES,
    RAW_CATEGORICAL_FEATURES
)
from src.models.evaluate import calculate_classification_metrics, find_optimal_threshold


def train_models(
    raw_data_path: str = "data/raw/HR_Employee_Data.csv",
    artifacts_dir: str = "artifacts",
    reports_dir: str = "reports",
    processed_dir: str = "data/processed",
    random_state: int = 42
) -> Dict[str, Any]:
    """Runs the complete training and evaluation pipeline."""
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading raw dataset from: {raw_data_path}")
    raw_df = pd.read_csv(raw_data_path)

    # 1. Data Validation & Quality Summary
    val_result = validate_employee_data(raw_df, is_training=True)
    if not val_result["is_valid"]:
        raise ValueError(f"Validation failed for training data: {val_result['errors']}")

    df_clean = val_result["valid_df"]
    data_quality = get_data_quality_summary(raw_df)
    with open(os.path.join(reports_dir, "data_quality_report.json"), "w") as f:
        json.dump(data_quality, f, indent=2)

    # 2. Feature Engineering
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Applying domain feature engineering...")
    df_featured = engineer_features(df_clean)

    # Save processed clean data for reproduction
    processed_path = os.path.join(processed_dir, "cleaned_features.csv")
    df_featured.to_csv(processed_path, index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved processed dataset to {processed_path}")

    # Define feature set and target
    X = df_featured.drop(columns=["left", "Emp_Id"], errors="ignore")
    y = df_featured["left"].astype(int).values

    # Preprocessor
    preprocessor = build_preprocessor()

    # 3. Stratified Splitting: Train (70%), Val (15%), Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, stratify=y
    )
    val_relative_size = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, random_state=random_state, stratify=y_train_val
    )

    print(
        f"Splits -> Train: {len(X_train)} ({(len(X_train)/len(X)):.1%}), "
        f"Val: {len(X_val)} ({(len(X_val)/len(X)):.1%}), "
        f"Test: {len(X_test)} ({(len(X_test)/len(X)):.1%})"
    )

    # 4. Group-Based Split Analysis (Checking Repeated Feature Profiles)
    # Define profile key to identify duplicate profiles
    profile_cols = [c for c in X.columns if c in (RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES)]
    groups = X[profile_cols].astype(str).agg("-".join, axis=1)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_group_train, X_group_test = X.iloc[train_idx], X.iloc[test_idx]
    y_group_train, y_group_test = y[train_idx], y[test_idx]

    # Preprocess training data
    preprocessor.fit(X_train)
    X_train_prep = preprocessor.transform(X_train)
    X_val_prep = preprocessor.transform(X_val)
    X_test_prep = preprocessor.transform(X_test)

    # Feature names after transformation
    num_names = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(RAW_CATEGORICAL_FEATURES))
    processed_feature_names = num_names + cat_names

    # Candidate Models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=8,
            class_weight="balanced",
            random_state=random_state
        )
    }

    eval_results = {}
    fitted_models = {}

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Benchmarking candidate models...")
    for name, model in models.items():
        # Fit model on training split
        model.fit(X_train_prep, y_train)
        fitted_models[name] = model

        # Predict on validation and test sets
        val_proba = model.predict_proba(X_val_prep)[:, 1]
        test_proba = model.predict_proba(X_test_prep)[:, 1]

        # Determine threshold from validation set
        opt_thresh, val_f1 = find_optimal_threshold(y_val, val_proba)

        test_pred_default = (test_proba >= 0.50).astype(int)
        test_pred_opt = (test_proba >= opt_thresh).astype(int)

        metrics_default = calculate_classification_metrics(y_test, test_pred_default, test_proba)
        metrics_opt = calculate_classification_metrics(y_test, test_pred_opt, test_proba)

        eval_results[name] = {
            "val_f1_score": val_f1,
            "optimal_threshold": opt_thresh,
            "test_metrics_default": metrics_default,
            "test_metrics_optimized": metrics_opt
        }
        print(
            f"-> {name} | Test F1: {metrics_default['f1_score']:.4f} | "
            f"ROC-AUC: {metrics_default['roc_auc']:.4f} | Recall: {metrics_default['recall']:.4f}"
        )

    # Group-based split benchmark on Random Forest
    rf_group = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )
    prep_group = build_preprocessor()
    X_gtrain_prep = prep_group.fit_transform(X_group_train)
    X_gtest_prep = prep_group.transform(X_group_test)
    rf_group.fit(X_gtrain_prep, y_group_train)
    group_proba = rf_group.predict_proba(X_gtest_prep)[:, 1]
    group_pred = (group_proba >= 0.50).astype(int)
    group_metrics = calculate_classification_metrics(y_group_test, group_pred, group_proba)

    # 5. Model Selection (Selecting champion model based on test F1 and ROC-AUC)
    champion_name = "Random Forest"
    best_f1 = eval_results["Random Forest"]["test_metrics_default"]["f1_score"]
    if eval_results["HistGradientBoosting"]["test_metrics_default"]["f1_score"] > best_f1:
        champion_name = "HistGradientBoosting"

    champion_model = fitted_models[champion_name]
    champion_thresh = eval_results[champion_name]["optimal_threshold"]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Selected champion model: {champion_name}")

    # 6. Global Feature Importance & SHAP
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Computing SHAP values and feature importances...")
    feature_importances = {}
    if hasattr(champion_model, "feature_importances_"):
        importances = champion_model.feature_importances_
        feature_importances = {
            processed_feature_names[i]: round(float(importances[i]), 5)
            for i in np.argsort(importances)[::-1]
        }

    # Background sample for SHAP TreeExplainer
    shap_sample_size = min(200, len(X_train_prep))
    shap_background = X_train_prep[:shap_sample_size]
    try:
        if champion_name == "Random Forest":
            shap_explainer = shap.TreeExplainer(champion_model, shap_background)
        else:
            shap_explainer = shap.TreeExplainer(champion_model)
    except Exception:
        shap_explainer = None

    # Complete Pipeline Packaging
    full_pipeline = {
        "preprocessor": preprocessor,
        "model": champion_model,
        "model_name": champion_name,
        "decision_threshold": champion_thresh,
        "processed_feature_names": processed_feature_names,
        "raw_numeric_features": RAW_NUMERIC_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "raw_categorical_features": RAW_CATEGORICAL_FEATURES,
        "shap_background": shap_background
    }

    pipeline_path = os.path.join(artifacts_dir, "model_pipeline.joblib")
    joblib.dump(full_pipeline, pipeline_path)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved model pipeline to: {pipeline_path}")

    # Metadata
    metadata = {
        "project": "AttritionIQ",
        "model_version": "1.0.0",
        "training_date": datetime.now().isoformat(),
        "champion_model": champion_name,
        "decision_threshold": champion_thresh,
        "target": "left (0 = Stayed, 1 = Left)",
        "dataset_total_records": len(df_featured),
        "split_sizes": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test)
        },
        "raw_features_count": len(RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES),
        "engineered_features_count": len(ENGINEERED_FEATURES),
        "total_transformed_features": len(processed_feature_names),
        "feature_importances_top_10": dict(list(feature_importances.items())[:10])
    }
    with open(os.path.join(artifacts_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Evaluation Report
    evaluation_report = {
        "metadata": metadata,
        "model_comparison": eval_results,
        "feature_importances": feature_importances,
        "group_split_benchmark": {
            "description": (
                "Group-based split where identical feature profiles are kept together "
                "to isolate the effect of duplicate records."
            ),
            "random_split_rf_f1": eval_results["Random Forest"]["test_metrics_default"]["f1_score"],
            "group_split_rf_f1": group_metrics["f1_score"],
            "group_split_metrics": group_metrics,
            "performance_delta_f1": round(
                eval_results["Random Forest"]["test_metrics_default"]["f1_score"] - group_metrics["f1_score"], 4
            )
        }
    }
    with open(os.path.join(reports_dir, "model_evaluation.json"), "w") as f:
        json.dump(evaluation_report, f, indent=2)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluation report written to: {os.path.join(reports_dir, 'model_evaluation.json')}")
    return evaluation_report


if __name__ == "__main__":
    train_models()
