"""Screen 5: What-If Scenario Lab & Counterfactual Policy Simulator for AttritionIQ.

Enables HR decision-makers to simulate workplace interventions—such as workload rebalancing
or compensation adjustments—and observe model-estimated flight risk variations.
"""

import streamlit as st
import pandas as pd

from components.sidebar import render_sidebar
from components.metrics import render_kpi_card, render_risk_badge, render_alert_box
from components.charts import create_what_if_comparison_chart
from src.models.predict import predict_single, load_pipeline

st.set_page_config(
    page_title="What-If Scenario Lab | AttritionIQ",
    page_icon="🧪",
    layout="wide"
)

render_sidebar()

st.title("🧪 What-If Scenario Lab")
st.markdown("Simulate HR policy interventions and examine counterfactual flight risk shifts in real time.")

# Load pipeline
try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Trained pipeline missing: {e}. Please train model first via `python -m src.models.train`.")
    st.stop()

# Causal Disclaimer Banner
render_alert_box(
    title="Counterfactual & Causal Modeling Disclaimer",
    text=(
        "This tool executes an in-silico mathematical simulation using the fitted machine learning model. "
        "A predicted probability reduction indicates a favorable statistical shift, but does not guarantee "
        "or prove an empirical causal outcome. Real-world talent interventions require qualitative alignment and context."
    ),
    alert_type="info"
)

# Baseline Profile Selection
st.subheader("1. Configure Baseline Employee Profile")
base_c1, base_c2, base_c3, base_c4 = st.columns(4)
with base_c1:
    base_sat = st.slider("Baseline Satisfaction", 0.05, 1.0, 0.35, 0.01)
    base_eval = st.slider("Baseline Evaluation", 0.1, 1.0, 0.85, 0.01)
with base_c2:
    base_hours = st.slider("Baseline Monthly Hours", 80, 320, 265, 5)
    base_proj = st.slider("Baseline Projects", 1, 10, 6, 1)
with base_c3:
    base_tenure = st.slider("Tenure (Years)", 1, 12, 4, 1)
    base_salary = st.selectbox("Baseline Salary Tier", ["low", "medium", "high"], index=0)
with base_c4:
    base_dept = st.selectbox("Department", ["sales", "technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting", "hr", "management"], index=1)
    base_promo = st.selectbox("Promoted Recently?", ["No", "Yes"], index=0)

base_record = {
    "satisfaction_level": base_sat,
    "last_evaluation": base_eval,
    "number_project": base_proj,
    "average_montly_hours": base_hours,
    "time_spend_company": base_tenure,
    "Work_accident": 0,
    "promotion_last_5years": 1 if base_promo == "Yes" else 0,
    "Department": base_dept,
    "salary": base_salary
}

# 2. Alternative Counterfactual Scenario
st.markdown("---")
st.subheader("2. Propose Counterfactual Interventions")
st.caption("Adjust one or more parameters to simulate the potential impact of organizational or workload interventions.")

alt_c1, alt_c2, alt_c3, alt_c4 = st.columns(4)
with alt_c1:
    alt_sat = st.slider("Alternative Satisfaction", 0.05, 1.0, min(1.0, base_sat + 0.30), 0.01)
with alt_c2:
    alt_hours = st.slider("Alternative Monthly Hours", 80, 320, max(80, base_hours - 50), 5)
with alt_c3:
    alt_proj = st.slider("Alternative Projects", 1, 10, max(1, base_proj - 2), 1)
with alt_c4:
    alt_salary = st.selectbox("Alternative Salary Tier", ["low", "medium", "high"], index=1 if base_salary == "low" else 2)

alt_record = {
    "satisfaction_level": alt_sat,
    "last_evaluation": base_eval,
    "number_project": alt_proj,
    "average_montly_hours": alt_hours,
    "time_spend_company": base_tenure,
    "Work_accident": 0,
    "promotion_last_5years": 1 if base_promo == "Yes" else 0,
    "Department": base_dept,
    "salary": alt_salary
}

# Run both predictions
base_pred = predict_single(base_record, pipeline=pipeline)
alt_pred = predict_single(alt_record, pipeline=pipeline)

base_prob = base_pred["probability"]
alt_prob = alt_pred["probability"]
prob_delta = alt_prob - base_prob
pct_delta = prob_delta * 100.0

st.markdown("---")
st.subheader("3. Simulation Outcome & Risk Shift")

k_c1, k_c2, k_c3 = st.columns(3)
with k_c1:
    render_kpi_card(
        "Baseline Estimated Risk",
        f"{base_pred['probability_pct']}%",
        subtext=f"Risk Tier: {base_pred['risk_category']}"
    )
with k_c2:
    render_kpi_card(
        "Simulated Alternative Risk",
        f"{alt_pred['probability_pct']}%",
        subtext=f"Risk Tier: {alt_pred['risk_category']}"
    )
with k_c3:
    delta_str = f"{pct_delta:+.1f}%"
    d_color = "positive" if pct_delta < 0 else ("negative" if pct_delta > 0 else "normal")
    render_kpi_card(
        "Net Probability Change",
        delta_str,
        delta="Risk Reduction" if pct_delta < 0 else "Risk Increase",
        delta_color=d_color,
        subtext="Simulated net impact"
    )

# Visual comparison
v_c1, v_c2 = st.columns([1, 1.2])
with v_c1:
    fig_comp = create_what_if_comparison_chart(base_prob, alt_prob)
    st.plotly_chart(fig_comp, use_container_width=True)

with v_c2:
    st.markdown("##### 🔍 Parameter Comparison Matrix")
    matrix_data = [
        {"Parameter": "Satisfaction Score", "Baseline Profile": f"{base_sat:.2f}", "Alternative Simulation": f"{alt_sat:.2f}", "Difference": f"{alt_sat - base_sat:+.2f}"},
        {"Parameter": "Average Monthly Hours", "Baseline Profile": f"{base_hours}h", "Alternative Simulation": f"{alt_hours}h", "Difference": f"{alt_hours - base_hours:+d}h"},
        {"Parameter": "Concurrent Projects", "Baseline Profile": f"{base_proj}", "Alternative Simulation": f"{alt_proj}", "Difference": f"{alt_proj - base_proj:+d}"},
        {"Parameter": "Salary Tier", "Baseline Profile": f"{base_salary.upper()}", "Alternative Simulation": f"{alt_salary.upper()}", "Difference": f"{base_salary} → {alt_salary}"},
        {"Parameter": "Flight Risk Probability", "Baseline Profile": f"{base_pred['probability_pct']}%", "Alternative Simulation": f"{alt_pred['probability_pct']}%", "Difference": f"{pct_delta:+.1f}%"},
        {"Parameter": "Risk Classification", "Baseline Profile": f"{base_pred['risk_category']}", "Alternative Simulation": f"{alt_pred['risk_category']}", "Difference": f"{base_pred['risk_category']} → {alt_pred['risk_category']}"}
    ]
    st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)
