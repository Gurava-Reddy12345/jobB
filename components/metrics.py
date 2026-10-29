"""Reusable KPI card, badge, and alert components for AttritionIQ."""

import textwrap
import streamlit as st


def _render_html(html_str: str) -> None:
    """Renders HTML cleanly using st.html if available, or dedented markdown."""
    clean = textwrap.dedent(html_str).strip()
    if hasattr(st, "html"):
        st.html(clean)
    else:
        st.markdown(clean, unsafe_allow_html=True)


def render_kpi_card(
    label: str,
    value: str,
    delta: str = "",
    delta_color: str = "normal",
    subtext: str = ""
) -> None:
    """Renders a styled SaaS metric card."""
    delta_html = ""
    if delta:
        color = "#10b981" if delta_color == "positive" else ("#ef4444" if delta_color == "negative" else "#64748b")
        delta_html = f"<div style='font-size: 0.82rem; color: {color}; font-weight: 600; margin-top: 4px;'>{delta}</div>"

    subtext_html = ""
    if subtext:
        subtext_html = f"<div style='font-size: 0.75rem; color: #64748b; margin-top: 2px;'>{subtext}</div>"

    card_html = (
        "<div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.1rem 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 0.8rem;'>"
        f"<div style='font-size: 0.78rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em;'>{label}</div>"
        f"<div style='font-size: 1.75rem; font-weight: 700; color: #0f172a; margin-top: 4px; line-height: 1.2;'>{value}</div>"
        f"{delta_html}"
        f"{subtext_html}"
        "</div>"
    )
    _render_html(card_html)


def render_risk_badge(category: str) -> str:
    """Returns HTML for a styled risk badge as an inline single-line string."""
    colors = {
        "Low Risk": {"bg": "#dcfce7", "text": "#166534", "border": "#86efac"},
        "Medium Risk": {"bg": "#fef3c7", "text": "#92400e", "border": "#fcd34d"},
        "High Risk": {"bg": "#fee2e2", "text": "#991b1b", "border": "#fca5a5"}
    }
    style = colors.get(category, {"bg": "#f1f5f9", "text": "#334155", "border": "#cbd5e1"})
    return f"<span style='background-color: {style['bg']}; color: {style['text']}; border: 1px solid {style['border']}; padding: 0.35rem 0.85rem; border-radius: 9999px; font-weight: 700; font-size: 0.88rem; display: inline-block;'>{category}</span>"


def render_alert_box(title: str, text: str, alert_type: str = "info") -> None:
    """Renders a styled notice/warning/reliability alert container."""
    styles = {
        "info": {"bg": "#eff6ff", "border": "#bfdbfe", "title": "#1e40af", "icon": "ℹ️"},
        "warning": {"bg": "#fffbeb", "border": "#fde68a", "title": "#b45309", "icon": "⚠️"},
        "danger": {"bg": "#fef2f2", "border": "#fecaca", "title": "#b91c1c", "icon": "🚨"},
        "success": {"bg": "#f0fdf4", "border": "#bbf7d0", "title": "#15803d", "icon": "✅"}
    }
    cfg = styles.get(alert_type, styles["info"])

    html = (
        f"<div style='background-color: {cfg['bg']}; border-left: 4px solid {cfg['border']}; border-radius: 8px; padding: 0.9rem 1.1rem; margin: 0.9rem 0;'>"
        f"<div style='font-weight: 700; color: {cfg['title']}; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;'>"
        f"<span>{cfg['icon']}</span> <span>{title}</span>"
        "</div>"
        f"<div style='color: #334155; font-size: 0.88rem; margin-top: 0.35rem; line-height: 1.45;'>{text}</div>"
        "</div>"
    )
    _render_html(html)
