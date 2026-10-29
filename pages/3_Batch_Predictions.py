"""Screen 3: Batch Predictions & Cohort Risk Scoring for AttritionIQ.

Enables bulk scoring of employee cohorts via CSV upload, enforces schema validation,
identifies invalid rows, and provides enriched CSV export with risk driver tags.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from components.sidebar import render_sidebar
from components.metrics import render_kpi_card, render_alert_box, render_risk_badge
from src.models.predict import predict_batch, load_pipeline

st.set_page_config(
    page_title="Batch Predictions | AttritionIQ",
    page_icon="📁",
    layout="wide"
)

render_sidebar()

st.title("📁 Batch Employee Predictions")
st.markdown("Upload new employee cohorts in CSV format, audit records, score flight probabilities, and download enriched analytics.")

# Pipeline verification
try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Trained pipeline missing: {e}. Please train model first via `python -m src.models.train`.")
    st.stop()

# Download Template Button
st.subheader("1. Template & Guidelines")
t_col1, t_col2 = st.columns([1, 2])
with t_col1:
    sample_path = os.path.join("data", "sample", "sample_new_employees.csv")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            st.download_button(
                label="📥 Download Sample CSV Template",
                data=f.read(),
                file_name="attrition_iq_sample_employees.csv",
                mime="text/csv",
                use_container_width=True
            )
with t_col2:
    st.caption(
        "**Required Schema:** satisfaction_level (0.0-1.0 or %), last_evaluation, number_project, "
        "average_montly_hours, time_spend_company, Work_accident (0/1), promotion_last_5years (0/1), "
        "Department, salary (low/medium/high). Target 'left' is strictly excluded to prevent leakage."
    )

st.markdown("---")

# CSV Uploader
st.subheader("2. Upload Cohort File")
uploaded_file = st.file_uploader(
    "Upload Employee Data CSV",
    type=["csv"],
    help="Select a CSV file containing employee records matching the expected schema."
)

if uploaded_file is not None:
    try:
        raw_uploaded_df = pd.read_csv(uploaded_file)
        st.write(f"**Loaded File:** `{uploaded_file.name}` ({len(raw_uploaded_df)} total records)")
    except Exception as e:
        st.error(f"Failed to parse CSV file: {e}")
        st.stop()

    # Run batch prediction and validation
    with st.spinner("Validating records and scoring cohort..."):
        batch_res = predict_batch(raw_uploaded_df, pipeline=pipeline)

    if batch_res.get("warnings"):
        for w in batch_res["warnings"]:
            st.warning(w)

    # If there are invalid rows, display them
    invalid_count = batch_res["invalid_count"]
    if invalid_count > 0:
        render_alert_box(
            title=f"Validation Alert: {invalid_count} Invalid Row(s) Detected",
            text="The records below failed validation checks (out of range, missing columns, or bad categories) and were excluded from scoring.",
            alert_type="warning"
        )
        invalid_display = batch_res["invalid_df"].copy()
        invalid_display["Error_Reason"] = batch_res["invalid_reasons"]
        with st.expander("View Invalid Records & Error Explanations"):
            st.dataframe(invalid_display, use_container_width=True)

    if not batch_res["success"] or batch_res["valid_count"] == 0:
        st.error("No valid records available for scoring. Please correct the invalid rows and re-upload.")
        st.stop()

    valid_df = batch_res["predictions_df"]

    # Cohort Risk Summary KPIs
    st.subheader("3. Cohort Flight Risk Summary")
    k_col1, k_col2, k_col3, k_col4 = st.columns(4)
    with k_col1:
        render_kpi_card("Total Scored", f"{len(valid_df):,}", subtext=f"Out of {len(raw_uploaded_df)} records")
    with k_col2:
        render_kpi_card("High Risk", f"{batch_res['high_risk_count']:,}", delta=f"{(batch_res['high_risk_count']/len(valid_df)*100):.1f}%", delta_color="negative")
    with k_col3:
        render_kpi_card("Medium Risk", f"{batch_res['medium_risk_count']:,}", delta=f"{(batch_res['medium_risk_count']/len(valid_df)*100):.1f}%", delta_color="normal")
    with k_col4:
        render_kpi_card("Low Risk", f"{batch_res['low_risk_count']:,}", delta=f"{(batch_res['low_risk_count']/len(valid_df)*100):.1f}%", delta_color="positive")

    # Cohort Risk Visuals
    v_col1, v_col2 = st.columns([1, 1.4])
    with v_col1:
        cat_counts = valid_df["risk_category"].value_counts().reset_index()
        cat_counts.columns = ["Risk Category", "Count"]
        color_map = {"Low Risk": "#10b981", "Medium Risk": "#f59e0b", "High Risk": "#ef4444"}
        fig_cat = px.pie(
            cat_counts,
            values="Count",
            names="Risk Category",
            color="Risk Category",
            color_discrete_map=color_map,
            hole=0.55,
            title="<b>Cohort Risk Tier Distribution</b>"
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    with v_col2:
        fig_hist = px.histogram(
            valid_df,
            x="estimated_risk_pct",
            nbins=20,
            title="<b>Flight Probability Distribution (%)</b>",
            color_discrete_sequence=["#1e3a8a"]
        )
        fig_hist.update_layout(xaxis_title="Flight Probability (%)", yaxis_title="Employee Count")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Results Table & CSV Export
    st.subheader("4. Scored Employee Records")

    # Filter by risk tier
    filter_tier = st.selectbox(
        "Filter by Risk Tier",
        ["All Tiers", "High Risk", "Medium Risk", "Low Risk"],
        index=0
    )
    display_df = valid_df.copy()
    if filter_tier != "All Tiers":
        display_df = display_df[display_df["risk_category"] == filter_tier]

    # Reorder columns nicely for display
    front_cols = ["Emp_Id", "risk_category", "estimated_risk_pct", "primary_risk_driver", "Department", "salary"]
    existing_front = [c for c in front_cols if c in display_df.columns]
    other_cols = [c for c in display_df.columns if c not in existing_front]
    ordered_cols = existing_front + other_cols

    st.dataframe(
        display_df[ordered_cols],
        use_container_width=True,
        hide_index=True
    )

    # CSV Download Button
    csv_bytes = valid_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Scored Batch Predictions (CSV)",
        data=csv_bytes,
        file_name="attrition_iq_predictions_scored.csv",
        mime="text/csv",
        type="primary"
    )

else:
    st.info("👆 Upload a CSV file or download the template above to start batch evaluation.")
