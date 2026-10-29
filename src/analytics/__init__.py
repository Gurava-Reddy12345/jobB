"""Workforce analytics and narrative module for AttritionIQ."""
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
from .narrative import (
    generate_llm_plain_english_summary,
    generate_deterministic_plain_english_summary,
    format_promotion_tenure_phrase
)

__all__ = [
    "load_and_prepare_historical_data",
    "calculate_kpis",
    "get_department_attrition_stats",
    "get_salary_attrition_stats",
    "get_tenure_attrition_stats",
    "get_hours_distribution_stats",
    "get_satisfaction_distribution_stats",
    "generate_workforce_insights",
    "generate_llm_plain_english_summary",
    "generate_deterministic_plain_english_summary",
    "format_promotion_tenure_phrase"
]
