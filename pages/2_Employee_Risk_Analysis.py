"""Screen 2: Individual Employee Risk Analysis & Explainability for AttritionIQ.

Accepts structured employee profile inputs, performs validated ML scoring,
displays SHAP attributions, and provides actionable HR retention recommendations.
"""

import streamlit as st
import pandas as pd

from components.sidebar import render_sidebar
from components.metrics import render_risk_badge, render_alert_box, render_kpi_card
from components.charts import create_risk_gauge, create_shap_waterfall_chart
from src.models.predict import predict_single, load_pipeline
from src.models.explain import explain_prediction
from src.analytics.narrative import (
    generate_llm_plain_english_summary,
    generate_deterministic_plain_english_summary
)

st.set_page_config(
    page_title="Employee Risk Analysis | AttritionIQ",
    page_icon="🎯",
    layout="wide"
)

render_sidebar()

st.title("🎯 Individual Employee Risk Analysis")
st.markdown("Assess flight risk probability for a specific employee profile with full SHAP transparency.")

# Check pipeline
try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Model pipeline not available: {e}. Please run `python -m src.models.train` first.")
    st.stop()

# Preset profiles for rapid evaluation
st.markdown("##### ⚡ Quick Scenario Presets")
preset_col1, preset_col2, preset_col3 = st.columns(3)

# Session state initialization for form inputs
if "p_sat" not in st.session_state:
    st.session_state["p_sat"] = 0.38
if "p_eval" not in st.session_state:
    st.session_state["p_eval"] = 0.85
if "p_proj" not in st.session_state:
    st.session_state["p_proj"] = 5
if "p_hours" not in st.session_state:
    st.session_state["p_hours"] = 260
if "p_tenure" not in st.session_state:
    st.session_state["p_tenure"] = 4
if "p_accident" not in st.session_state:
    st.session_state["p_accident"] = "No"
if "p_promo" not in st.session_state:
    st.session_state["p_promo"] = "No"
if "p_dept" not in st.session_state:
    st.session_state["p_dept"] = "sales"
if "p_salary" not in st.session_state:
    st.session_state["p_salary"] = "low"

with preset_col1:
    if st.button("🔥 Overworked Burnout Risk", use_container_width=True):
        st.session_state["p_sat"] = 0.12
        st.session_state["p_eval"] = 0.88
        st.session_state["p_proj"] = 6
        st.session_state["p_hours"] = 275
        st.session_state["p_tenure"] = 4
        st.session_state["p_accident"] = "No"
        st.session_state["p_promo"] = "No"
        st.session_state["p_dept"] = "technical"
        st.session_state["p_salary"] = "low"
        st.rerun()

with preset_col2:
    if st.button("🌟 Stable Retained Performer", use_container_width=True):
        st.session_state["p_sat"] = 0.84
        st.session_state["p_eval"] = 0.82
        st.session_state["p_proj"] = 4
        st.session_state["p_hours"] = 195
        st.session_state["p_tenure"] = 3
        st.session_state["p_accident"] = "No"
        st.session_state["p_promo"] = "No"
        st.session_state["p_dept"] = "management"
        st.session_state["p_salary"] = "high"
        st.rerun()

with preset_col3:
    if st.button("📉 Disengaged Contributor", use_container_width=True):
        st.session_state["p_sat"] = 0.35
        st.session_state["p_eval"] = 0.52
        st.session_state["p_proj"] = 2
        st.session_state["p_hours"] = 145
        st.session_state["p_tenure"] = 3
        st.session_state["p_accident"] = "No"
        st.session_state["p_promo"] = "No"
        st.session_state["p_dept"] = "support"
        st.session_state["p_salary"] = "medium"
        st.rerun()

st.write("")

# Form layout
form_col, result_col = st.columns([1.1, 1.2])

with form_col:
    st.subheader("Employee Attributes")
    with st.form("employee_profile_form"):
        f_c1, f_c2 = st.columns(2)
        with f_c1:
            satisfaction = st.slider(
                "Satisfaction Score (0.0 to 1.0)",
                min_value=0.01,
                max_value=1.00,
                value=float(st.session_state["p_sat"]),
                step=0.01,
                help="Proportion between 0.00 and 1.00 (e.g., 0.38 equals 38%)."
            )
            projects = st.slider(
                "Number of Assigned Projects",
                min_value=1,
                max_value=10,
                value=int(st.session_state["p_proj"]),
                step=1
            )
            tenure = st.slider(
                "Tenure at Company (Years)",
                min_value=1,
                max_value=15,
                value=int(st.session_state["p_tenure"]),
                step=1
            )
            promo = st.selectbox(
                "Promoted in Last 5 Years?",
                ["No", "Yes"],
                index=0 if st.session_state["p_promo"] == "No" else 1
            )
            dept = st.selectbox(
                "Department",
                [
                    "sales", "technical", "support", "IT", "product_mng",
                    "marketing", "RandD", "accounting", "hr", "management"
                ],
                index=[
                    "sales", "technical", "support", "IT", "product_mng",
                    "marketing", "RandD", "accounting", "hr", "management"
                ].index(st.session_state["p_dept"])
            )

        with f_c2:
            evaluation = st.slider(
                "Last Performance Evaluation (0.0 to 1.0)",
                min_value=0.01,
                max_value=1.00,
                value=float(st.session_state["p_eval"]),
                step=0.01,
                help="Evaluation score rating from 0.00 to 1.00."
            )
            hours = st.slider(
                "Average Monthly Working Hours",
                min_value=60,
                max_value=330,
                value=int(st.session_state["p_hours"]),
                step=5
            )
            accident = st.selectbox(
                "Work Accident on Record?",
                ["No", "Yes"],
                index=0 if st.session_state["p_accident"] == "No" else 1
            )
            salary = st.selectbox(
                "Salary Band",
                ["low", "medium", "high"],
                index=["low", "medium", "high"].index(st.session_state["p_salary"])
            )

        submit_btn = st.form_submit_button("⚡ Predict Attrition Risk", use_container_width=True, type="primary")

with result_col:
    st.subheader("Inference & Assessment")

    record = {
        "satisfaction_level": satisfaction,
        "last_evaluation": evaluation,
        "number_project": projects,
        "average_montly_hours": hours,
        "time_spend_company": tenure,
        "Work_accident": 1 if accident == "Yes" else 0,
        "promotion_last_5years": 1 if promo == "Yes" else 0,
        "Department": dept,
        "salary": salary
    }

    # Run prediction
    pred_res = predict_single(record, pipeline=pipeline)

    if not pred_res["success"]:
        st.error(f"Prediction error: {pred_res['errors']}")
    else:
        prob = pred_res["probability"]
        prob_pct = pred_res["probability_pct"]
        cat = pred_res["risk_category"]

        # Gauge & Category
        g_c1, g_c2 = st.columns([1.1, 0.9])
        with g_c1:
            fig_gauge = create_risk_gauge(prob)
            st.plotly_chart(fig_gauge, use_container_width=True)
        with g_c2:
            st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
            st.markdown(f"**Risk Classification:** {render_risk_badge(cat)}", unsafe_allow_html=True)
            st.markdown(f"**Decision Threshold:** `{pred_res['decision_threshold']:.2f}`")
            st.markdown(f"**Model Champion:** `{pipeline.get('model_name', 'Random Forest')}`")
            st.markdown(f"**Estimated Flight Likelihood:** `{prob_pct}%`")

        # Plain-English Executive HR Narrative (For Non-Technical Users)
        st.markdown("---")
        st.subheader("🗣️ Plain-English Executive HR Summary")
        st.caption("A human-centered breakdown translated for non-technical managers and business leaders.")

        cache_key = f"narrative_{prob}_{record['satisfaction_level']}_{record['last_evaluation']}_{record['time_spend_company']}_{record['average_montly_hours']}_{record['number_project']}"

        if cache_key not in st.session_state:
            with st.spinner("Drafting executive HR summary with OpenRouter AI..."):
                st.session_state[cache_key] = generate_llm_plain_english_summary(
                    record=record,
                    probability=prob,
                    risk_category=cat
                )

        narrative_text = st.session_state[cache_key]

        st.markdown(
            f"""
            <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #1e3a8a; border-radius: 10px; padding: 1.25rem 1.4rem; margin: 0.8rem 0; font-size: 0.95rem; line-height: 1.65; color: #1e293b; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                {narrative_text.replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True
        )

        n_col1, n_col2 = st.columns(2)
        with n_col1:
            if st.button("🔄 Regenerate with AI (OpenRouter LLM)", use_container_width=True):
                with st.spinner("Querying OpenRouter LLM for refreshed narrative..."):
                    st.session_state[cache_key] = generate_llm_plain_english_summary(
                        record=record,
                        probability=prob,
                        risk_category=cat
                    )
                st.rerun()
        with n_col2:
            if st.button("📝 Standard Rule-Based Narrative", use_container_width=True):
                st.session_state[cache_key] = generate_deterministic_plain_english_summary(
                    record=record,
                    probability=prob,
                    risk_category=cat
                )
                st.rerun()

        st.markdown("---")
        st.subheader("🔍 Algorithmic Explainability & Drivers")

        # Explainability SHAP chart
        with st.spinner("Calculating SHAP feature attributions..."):
            explanation = explain_prediction(record, pipeline=pipeline)

        fig_shap = create_shap_waterfall_chart(explanation["top_contributions"], explanation["base_value"])
        st.plotly_chart(fig_shap, use_container_width=True)

        # Recommended Review Areas
        st.markdown("##### 📋 Recommended HR Review Actions")
        for act in pred_res["recommendations"]:
            st.info(act)

        # Model Limitations & Audit Trail Box
        with st.expander("🛡️ Prediction Audit Trail & Methodological Limitations"):
            st.json(pred_res["audit"])
            st.caption(
                "Model associations reflect historical patterns and should be treated as diagnostic aids. "
                "They do not provide clinical or causal proof of employee departure."
            )
