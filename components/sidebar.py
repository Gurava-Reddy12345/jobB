"""Sidebar navigation and branding component for AttritionIQ."""

import os
import textwrap
import streamlit as st


def _render_html(html_str: str) -> None:
    clean = textwrap.dedent(html_str).strip()
    if hasattr(st, "html"):
        st.html(clean)
    else:
        st.markdown(clean, unsafe_allow_html=True)


def render_sidebar():
    """Renders the executive sidebar brand, status, and navigation aid."""
    with st.sidebar:
        brand_html = (
            "<div style='padding: 10px 0 20px 0;'>"
            "<div style='display: flex; align-items: center; gap: 10px;'>"
            "<div style='background: #1e3a8a; color: white; width: 38px; height: 38px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.2rem;'>"
            "IQ"
            "</div>"
            "<div>"
            "<div style='font-size: 1.25rem; font-weight: 800; color: #0f172a; line-height: 1.1;'>AttritionIQ</div>"
            "<div style='font-size: 0.75rem; color: #64748b; font-weight: 500;'>Workforce Intelligence</div>"
            "</div>"
            "</div>"
            "</div>"
        )
        _render_html(brand_html)

        st.markdown("---")

        # Model Status badge (checking both relative and absolute paths)
        root_dir = os.path.dirname(os.path.dirname(__file__))
        alt_path = os.path.join(root_dir, "artifacts", "model_pipeline.joblib")
        pipeline_exists = os.path.exists("artifacts/model_pipeline.joblib") or os.path.exists(alt_path)

        if pipeline_exists:
            st.success("🟢 Model Pipeline: Active (v1.0)")
        else:
            st.warning("🟡 Model Pipeline: Untrained")

        st.caption("Historical Dataset: 14,999 records")

        st.markdown("---")
        st.markdown("### Navigation Modules")
        st.markdown(
            """
- 📊 **Workforce Overview**
- 🎯 **Employee Risk Analysis**
- 📁 **Batch Predictions**
- 📈 **Model Performance**
- 🧪 **What-If Scenario Lab**
- 🛡️ **Data Quality & Audit**
            """
        )

        st.markdown("---")
        notice_html = (
            "<div style='font-size: 0.72rem; color: #94a3b8; line-height: 1.4;'>"
            "<b>Platform Notice</b><br>"
            "Predictions reflect statistical associations from historical benchmarks. "
            "Grounded decisions require holistic human-in-the-loop review."
            "</div>"
        )
        _render_html(notice_html)
