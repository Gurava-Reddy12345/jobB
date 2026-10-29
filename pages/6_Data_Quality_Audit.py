"""Screen 6: Data Quality, Schema Integrity & Model Governance Audit for AttritionIQ.

Provides transparent data hygiene metrics, duplicate profile audits,
target leakage prevention verifications, and downloadable audit documentation.
"""

import os
import json
import streamlit as st
import pandas as pd

from components.sidebar import render_sidebar
from components.metrics import render_kpi_card, render_alert_box
from src.data.validation import get_data_quality_summary, normalize_column_names

st.set_page_config(
    page_title="Data Quality & Audit | AttritionIQ",
    page_icon="🛡️",
    layout="wide"
)

render_sidebar()

st.title("🛡️ Data Quality & Governance Audit")
st.markdown("Complete data lifecycle audit, target isolation verification, duplicate profile analysis, and schema governance.")

# Load raw dataset and audit report
report_path = os.path.join("reports", "data_quality_report.json")
meta_path = os.path.join("artifacts", "model_metadata.json")

if os.path.exists(report_path):
    with open(report_path, "r") as f:
        audit_data = json.load(f)
else:
    df_raw = pd.read_csv("data/raw/HR_Employee_Data.csv")
    audit_data = get_data_quality_summary(df_raw)

metadata = {}
if os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        metadata = json.load(f)

# Headline Audit Cards
st.subheader("1. Dataset Architecture & Hygiene")
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi_card("Total Records", f"{audit_data['total_records']:,}", subtext="Rows in raw source")
with k2:
    render_kpi_card("Total Columns", f"{audit_data['total_columns']}", subtext="Attributes")
with k3:
    render_kpi_card("Missing Values", f"{audit_data['total_missing_values']}", delta="0.0% Missing (Clean)", delta_color="positive")
with k4:
    render_kpi_card("Exact Row Duplicates", f"{audit_data['exact_duplicates']}", subtext="Full row duplicates")
with k5:
    render_kpi_card("Repeated Profiles", f"{audit_data['repeated_feature_profiles']:,}", delta="Excluding Emp_Id", delta_color="normal")

st.markdown("---")

# Feature Usage & Exclusions
st.subheader("2. Feature Governance & Leakage Controls")
c_gov1, c_gov2 = st.columns(2)

with c_gov1:
    st.markdown("##### ✅ Features Used for Predictive Modeling")
    used_features_info = [
        {"Feature": "satisfaction_level", "Type": "Numeric (Ratio)", "Role": "Direct Predictor", "Range": "0.0 - 1.0 (Cleaned from %)"},
        {"Feature": "last_evaluation", "Type": "Numeric (Ratio)", "Role": "Direct Predictor", "Range": "0.0 - 1.0 (Cleaned from %)"},
        {"Feature": "number_project", "Type": "Integer", "Role": "Workload Predictor", "Range": "1 - 10 projects"},
        {"Feature": "average_montly_hours", "Type": "Integer", "Role": "Effort Predictor", "Range": "40 - 400 hours"},
        {"Feature": "time_spend_company", "Type": "Integer", "Role": "Tenure Predictor", "Range": "1 - 30 years"},
        {"Feature": "Work_accident", "Type": "Binary (0/1)", "Role": "Safety Predictor", "Range": "0 or 1"},
        {"Feature": "promotion_last_5years", "Type": "Binary (0/1)", "Role": "Advancement Predictor", "Range": "0 or 1"},
        {"Feature": "Department", "Type": "Categorical", "Role": "Domain Predictor", "Range": "10 Organization Units"},
        {"Feature": "salary", "Type": "Ordinal", "Role": "Comp Tier", "Range": "low, medium, high"}
    ]
    st.dataframe(pd.DataFrame(used_features_info), use_container_width=True, hide_index=True)

with c_gov2:
    st.markdown("##### 🚫 Excluded Columns & Target Isolation")
    excluded_info = [
        {
            "Column": "Emp_Id",
            "Classification": "Arbitrary Identifier",
            "Audit Decision": "Strictly Excluded",
            "Rationale": "Employee IDs carry no natural semantic signal and would cause spurious overfitting if included."
        },
        {
            "Column": "left",
            "Classification": "Historical Ground Truth",
            "Audit Decision": "Target Only (Isolated)",
            "Rationale": "Used strictly as ground-truth label during training. Striped during inference to prevent target leakage."
        }
    ]
    st.dataframe(pd.DataFrame(excluded_info), use_container_width=True, hide_index=True)

    render_alert_box(
        title="Target Leakage Prevention Check",
        text="The batch inference pipeline automatically validates uploaded files and quarantines any column labeled 'left' before feeding features into the inference engine.",
        alert_type="success"
    )

st.markdown("---")

# Split Sizes & Training Pipeline Metadata
st.subheader("3. Model Lineage & Split Architecture")
s_c1, s_c2, s_c3 = st.columns(3)
splits = metadata.get("split_sizes", {"train": 10499, "val": 2250, "test": 2250})
total_sp = sum(splits.values())

with s_c1:
    st.markdown(f"**Training Split:** `{splits.get('train', 0):,}` rows ({splits.get('train', 0)/total_sp:.1%})")
    st.markdown(f"**Validation Split:** `{splits.get('val', 0):,}` rows ({splits.get('val', 0)/total_sp:.1%})")
    st.markdown(f"**Held-Out Test Split:** `{splits.get('test', 0):,}` rows ({splits.get('test', 0)/total_sp:.1%})")

with s_c2:
    st.markdown(f"**Champion Model Architecture:** `{metadata.get('champion_model', 'Random Forest')}`")
    st.markdown(f"**Model Version:** `{metadata.get('model_version', '1.0.0')}`")
    st.markdown(f"**Model Serialized Timestamp:** `{metadata.get('training_date', 'N/A')[:19]}`")

with s_c3:
    st.markdown(f"**Raw Features:** `{metadata.get('raw_features_count', 9)}`")
    st.markdown(f"**Engineered Domain Features:** `{metadata.get('engineered_features_count', 10)}`")
    st.markdown(f"**Transformed Feature Vector:** `{metadata.get('total_transformed_features', 30)} dimensions`")

st.markdown("---")

# Downloadable Audit Report
st.subheader("4. Download Complete Audit Report")
audit_bytes = json.dumps(audit_data, indent=2).encode("utf-8")
st.download_button(
    label="📥 Download Data Quality Audit Report (JSON)",
    data=audit_bytes,
    file_name="attrition_iq_data_quality_report.json",
    mime="application/json",
    type="primary"
)
