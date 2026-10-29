"""Workforce analytics module for AttritionIQ."""
from .workforce import (
    load_and_prepare_historical_data,
    calculate_kpis,
    get_department_attrition_stats,
    get_salary_attrition_stats,
    get_tenure_attrition_stats,
    get_hours_distribution_stats,
    get_satisfaction_distribution_stats,
    generate_workforce_insights
)

__all__ = [
    "load_and_prepare_historical_data",
    "calculate_kpis",
    "get_department_attrition_stats",
    "get_salary_attrition_stats",
    "get_tenure_attrition_stats",
    "get_hours_distribution_stats",
    "get_satisfaction_distribution_stats",
    "generate_workforce_insights"
]
