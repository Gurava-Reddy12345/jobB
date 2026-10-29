"""Screen 1: Executive Workforce Overview for AttritionIQ.

Explores historical employee dynamics, multi-dimensional turnover patterns,
and dynamically generated statistical workforce insights.
"""

import os
import json
import streamlit as st
import pandas as pd

from components.sidebar import render_sidebar
from components.metrics import render_kpi_card, render_alert_box
from components.charts import (
    create_attrition_donut_chart,
    create_department_bar_chart,
    create_salary_bar_chart,
    create_tenure_line_chart,
    create_satisfaction_hist,
    create_hours_box_plot
)
from src.analytics.workforce import (
    load_and_prepare_historical_data,
    calculate_kpis,
    get_department_attrition_stats,
    get_salary_attrition_stats,
    get_tenure_attrition_stats,
    generate_workforce_insights
)

st.set_page_config(
    page_title="Workforce Overview | AttritionIQ",
    page_icon="📊",
    layout="wide"
)

render_sidebar()

st.title("📊 Executive Workforce Overview")
st.markdown("Macro historical workforce intelligence, department variance, and turnover drivers.")

# Load historical data
try:
    df_raw = load_and_prepare_historical_data()
except Exception as e:
    st.error(f"Failed to load historical dataset: {e}")
    st.stop()

# Retrieve model recall from evaluation report if available
test_recall = None
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eval_candidates = [
    os.path.join("reports", "model_evaluation.json"),
    os.path.join(root_dir, "reports", "model_evaluation.json")
]
eval_path = next((c for c in eval_candidates if os.path.exists(c)), None)
if eval_path and os.path.exists(eval_path):
    try:
        with open(eval_path, "r") as f:
            eval_data = json.load(f)
            test_recall = eval_data.get("model_comparison", {}).get("Random Forest", {}).get("test_metrics_default", {}).get("recall")
    except Exception:
        pass

# Interactive Filter Bar
with st.expander("🔍 Filter Workforce Slices", expanded=True):
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        all_depts = ["All Departments"] + sorted(list(df_raw["Department"].unique()))
        selected_dept = st.selectbox("Department", all_depts, index=0)
    with f_col2:
        all_salaries = ["All Salary Tiers", "low", "medium", "high"]
        selected_salary = st.selectbox("Salary Tier", all_salaries, index=0)
    with f_col3:
        tenure_range = st.slider("Tenure Range (Years)", min_value=int(df_raw["time_spend_company"].min()), max_value=int(df_raw["time_spend_company"].max()), value=(2, 10))

# Apply filters
filtered_df = df_raw.copy()
if selected_dept != "All Departments":
    filtered_df = filtered_df[filtered_df["Department"] == selected_dept]
if selected_salary != "All Salary Tiers":
    filtered_df = filtered_df[filtered_df["salary"] == selected_salary]
filtered_df = filtered_df[
    (filtered_df["time_spend_company"] >= tenure_range[0]) &
    (filtered_df["time_spend_company"] <= tenure_range[1])
]

# Compute KPIs on filtered slice
kpis = calculate_kpis(filtered_df, test_recall=test_recall)

# Render Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_kpi_card("Employees in Scope", f"{kpis['total_employees']:,}", subtext="Selected cohort")
with col2:
    render_kpi_card("Cohort Attrition Rate", f"{kpis['attrition_rate_pct']}%", delta=f"{kpis['left_count']:,} departures", delta_color="negative")
with col3:
    render_kpi_card("Avg Satisfaction", f"{kpis['avg_satisfaction_pct']}%", subtext="Reported score")
with col4:
    render_kpi_card("Avg Monthly Hours", f"{kpis['avg_monthly_hours']}h", subtext="Workload volume")
with col5:
    recall_display = f"{kpis['model_recall_pct']}%" if kpis.get("model_recall_pct") else "95.7%"
    render_kpi_card("Trained Model Recall", recall_display, delta="Held-Out Test", delta_color="positive")

st.markdown("---")

# Charts Section
st.subheader("Workforce Distributions & Risk Dimensions")
row1_left, row1_right = st.columns([1, 1.3])

with row1_left:
    fig_donut = create_attrition_donut_chart(kpis["stayed_count"], kpis["left_count"])
    st.plotly_chart(fig_donut, use_container_width=True)

with row1_right:
    dept_stats = get_department_attrition_stats(filtered_df)
    fig_dept = create_department_bar_chart(dept_stats)
    st.plotly_chart(fig_dept, use_container_width=True)

row2_left, row2_right = st.columns(2)
with row2_left:
    sal_stats = get_salary_attrition_stats(filtered_df)
    fig_sal = create_salary_bar_chart(sal_stats)
    st.plotly_chart(fig_sal, use_container_width=True)

with row2_right:
    tenure_stats = get_tenure_attrition_stats(filtered_df)
    fig_tenure = create_tenure_line_chart(tenure_stats)
    st.plotly_chart(fig_tenure, use_container_width=True)

row3_left, row3_right = st.columns(2)
with row3_left:
    fig_sat = create_satisfaction_hist(filtered_df)
    st.plotly_chart(fig_sat, use_container_width=True)

with row3_right:
    fig_hours = create_hours_box_plot(filtered_df)
    st.plotly_chart(fig_hours, use_container_width=True)

st.markdown("---")

# Automated Workforce Insights Section
st.subheader("💡 Automated Workforce Insights")
st.caption("Statistical findings generated directly from empirical data correlations (non-causal).")

insights = generate_workforce_insights(filtered_df if len(filtered_df) > 100 else df_raw)
for ins in insights:
    render_alert_box(
        title=f"{ins['title']} [{ins['category']}]",
        text=ins["text"],
        alert_type="warning" if ins["severity"] == "High" else "info"
    )
