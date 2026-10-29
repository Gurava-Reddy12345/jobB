"""Interactive Plotly chart generation for AttritionIQ.

Curated color schemes, sleek typography, clean layouts, and tooltips
designed for executive presentation and technical interpretability.
"""

from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Theme palette
COLOR_PRIMARY = "#1e3a8a"    # Navy
COLOR_SECONDARY = "#0284c7"  # Sky
COLOR_ACCENT = "#f59e0b"     # Amber
COLOR_DANGER = "#ef4444"     # Red
COLOR_SUCCESS = "#10b981"    # Green
COLOR_MUTED = "#64748b"      # Slate

CHART_LAYOUT = dict(
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", color="#1e293b"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=40, b=20)
)


def create_attrition_donut_chart(stayed: int, left: int) -> go.Figure:
    """Donut chart showing workforce distribution."""
    fig = go.Figure(data=[
        go.Pie(
            labels=["Retained (Stayed)", "Attrition (Left)"],
            values=[stayed, left],
            hole=0.62,
            marker=dict(colors=[COLOR_SUCCESS, COLOR_DANGER]),
            textinfo="label+percent",
            hoverinfo="label+value+percent"
        )
    ])
    fig.update_layout(
        **CHART_LAYOUT,
        showlegend=False,
        title=dict(text="<b>Workforce Retention vs Attrition</b>", font=dict(size=15)),
        height=320
    )
    return fig


def create_department_bar_chart(dept_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart for department attrition rates."""
    df_sorted = dept_df.sort_values(by="attrition_rate_pct", ascending=True)
    fig = go.Figure(go.Bar(
        x=df_sorted["attrition_rate_pct"],
        y=df_sorted["Department"],
        orientation="h",
        marker=dict(
            color=df_sorted["attrition_rate_pct"],
            colorscale=[[0, "#93c5fd"], [1, "#1e3a8a"]],
            showscale=False
        ),
        text=[f"{v:.1f}%" for v in df_sorted["attrition_rate_pct"]],
        textposition="outside"
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>Attrition Rate by Department (%)</b>", font=dict(size=15)),
        xaxis=dict(title="Attrition Rate (%)", gridcolor="#f1f5f9"),
        yaxis=dict(title="", tickfont=dict(size=12)),
        height=380
    )
    return fig


def create_salary_bar_chart(sal_df: pd.DataFrame) -> go.Figure:
    """Bar chart for salary band attrition rates."""
    fig = go.Figure(go.Bar(
        x=sal_df["salary"].str.capitalize(),
        y=sal_df["attrition_rate_pct"],
        marker=dict(color=[COLOR_DANGER, COLOR_ACCENT, COLOR_SUCCESS]),
        text=[f"{v:.1f}%" for v in sal_df["attrition_rate_pct"]],
        textposition="outside"
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>Attrition Rate by Salary Band</b>", font=dict(size=15)),
        xaxis=dict(title="Salary Tier"),
        yaxis=dict(title="Attrition Rate (%)", gridcolor="#f1f5f9"),
        height=320
    )
    return fig


def create_tenure_line_chart(tenure_df: pd.DataFrame) -> go.Figure:
    """Trend line of attrition rate across company tenure."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tenure_df["time_spend_company"],
        y=tenure_df["attrition_rate_pct"],
        mode="lines+markers",
        name="Attrition Rate",
        line=dict(color=COLOR_DANGER, width=3),
        marker=dict(size=8, color=COLOR_DANGER)
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>Attrition Rate across Years at Company</b>", font=dict(size=15)),
        xaxis=dict(title="Years at Company (Tenure)", dtick=1, gridcolor="#f1f5f9"),
        yaxis=dict(title="Attrition Rate (%)", gridcolor="#f1f5f9"),
        height=320
    )
    return fig


def create_satisfaction_hist(df: pd.DataFrame) -> go.Figure:
    """Overlaid distribution of satisfaction level for stayed vs left."""
    fig = go.Figure()
    stayed_sat = df[df["left"] == 0]["satisfaction_level"] * 100.0
    left_sat = df[df["left"] == 1]["satisfaction_level"] * 100.0

    fig.add_trace(go.Histogram(
        x=stayed_sat,
        name="Stayed",
        opacity=0.65,
        marker=dict(color=COLOR_SUCCESS),
        nbinsx=30
    ))
    fig.add_trace(go.Histogram(
        x=left_sat,
        name="Left",
        opacity=0.75,
        marker=dict(color=COLOR_DANGER),
        nbinsx=30
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        barmode="overlay",
        title=dict(text="<b>Satisfaction Score Distribution (Stayed vs Left)</b>", font=dict(size=15)),
        xaxis=dict(title="Satisfaction Score (%)", gridcolor="#f1f5f9"),
        yaxis=dict(title="Headcount", gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=340
    )
    return fig


def create_hours_box_plot(df: pd.DataFrame) -> go.Figure:
    """Box plot of monthly hours by attrition status."""
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=df[df["left"] == 0]["average_montly_hours"],
        name="Stayed",
        marker=dict(color=COLOR_SUCCESS),
        boxpoints="outliers"
    ))
    fig.add_trace(go.Box(
        y=df[df["left"] == 1]["average_montly_hours"],
        name="Left",
        marker=dict(color=COLOR_DANGER),
        boxpoints="outliers"
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>Monthly Working Hours Distribution</b>", font=dict(size=15)),
        yaxis=dict(title="Average Monthly Hours", gridcolor="#f1f5f9"),
        height=340
    )
    return fig


def create_risk_gauge(probability: float) -> go.Figure:
    """Speedometer-style gauge indicating employee flight probability."""
    prob_pct = probability * 100.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number=dict(suffix="%", font=dict(size=36, color="#0f172a")),
        title=dict(text="<b>Estimated Flight Risk</b>", font=dict(size=15, color="#475569")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#94a3b8"),
            bar=dict(color="#0f172a", width=6),
            steps=[
                {"range": [0, 35], "color": "#dcfce7"},
                {"range": [35, 65], "color": "#fef3c7"},
                {"range": [65, 100], "color": "#fee2e2"}
            ],
            threshold=dict(
                line=dict(color="#b91c1c", width=3),
                thickness=0.85,
                value=prob_pct
            )
        )
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=260,
        margin=dict(l=30, r=30, t=50, b=10)
    )
    return fig


def create_shap_waterfall_chart(contributions: List[Dict[str, Any]], base_value: float) -> go.Figure:
    """Waterfall chart visualizing SHAP feature contributions for a single prediction."""
    features = [c["display_name"] for c in contributions]
    values = [c["shap_value"] for c in contributions]
    colors = [COLOR_DANGER if v > 0 else COLOR_SUCCESS for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.3f}" for v in values],
        textposition="outside"
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(
            text="<b>SHAP Feature Contribution (Impact on Flight Risk)</b>",
            font=dict(size=14)
        ),
        xaxis=dict(
            title="SHAP Value (Positive = Increases Risk, Negative = Decreases Risk)",
            gridcolor="#f1f5f9",
            zerolinecolor="#94a3b8",
            zerolinewidth=1.5
        ),
        yaxis=dict(autorange="reversed"),
        height=360
    )
    return fig


def create_feature_importance_bar(importances_dict: Dict[str, float], top_n: int = 10) -> go.Figure:
    """Horizontal bar chart for global feature importances."""
    items = list(importances_dict.items())[:top_n]
    items.reverse()  # Top at top
    labels = [k.replace("_", " ").title() for k, _ in items]
    vals = [v for _, v in items]

    fig = go.Figure(go.Bar(
        x=vals,
        y=labels,
        orientation="h",
        marker=dict(color=COLOR_PRIMARY),
        text=[f"{v:.3f}" for v in vals],
        textposition="outside"
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"<b>Top {top_n} Global Feature Importances</b>", font=dict(size=15)),
        xaxis=dict(title="Importance Weight", gridcolor="#f1f5f9"),
        height=380
    )
    return fig


def create_confusion_matrix_heatmap(cm: List[List[int]]) -> go.Figure:
    """Confusion matrix heatmap."""
    labels = ["Stayed (0)", "Left (1)"]
    z = cm
    fig = px.imshow(
        z,
        text_auto=True,
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>Held-Out Test Confusion Matrix</b>", font=dict(size=15)),
        height=320
    )
    return fig


def create_what_if_comparison_chart(
    base_prob: float,
    new_prob: float
) -> go.Figure:
    """Comparison bar chart between original profile and counterfactual scenario."""
    labels = ["Original Profile", "Alternative (What-If) Scenario"]
    probs = [base_prob * 100.0, new_prob * 100.0]
    colors = [COLOR_MUTED, COLOR_PRIMARY]

    fig = go.Figure(go.Bar(
        x=labels,
        y=probs,
        marker=dict(color=colors),
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
        width=0.45
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>Flight Probability Shift</b>", font=dict(size=15)),
        yaxis=dict(title="Estimated Risk (%)", range=[0, 105], gridcolor="#f1f5f9"),
        height=320
    )
    return fig
