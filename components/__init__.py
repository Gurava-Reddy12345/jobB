"""UI components package for AttritionIQ."""
from .sidebar import render_sidebar
from .metrics import render_kpi_card, render_risk_badge, render_alert_box
from .charts import (
    create_attrition_donut_chart,
    create_department_bar_chart,
    create_salary_bar_chart,
    create_tenure_line_chart,
    create_satisfaction_hist,
    create_hours_box_plot,
    create_risk_gauge,
    create_shap_waterfall_chart,
    create_feature_importance_bar,
    create_confusion_matrix_heatmap,
    create_what_if_comparison_chart
)

__all__ = [
    "render_sidebar",
    "render_kpi_card",
    "render_risk_badge",
    "render_alert_box",
    "create_attrition_donut_chart",
    "create_department_bar_chart",
    "create_salary_bar_chart",
    "create_tenure_line_chart",
    "create_satisfaction_hist",
    "create_hours_box_plot",
    "create_risk_gauge",
    "create_shap_waterfall_chart",
    "create_feature_importance_bar",
    "create_confusion_matrix_heatmap",
    "create_what_if_comparison_chart"
]
