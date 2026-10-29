"""Historical workforce analytics and automated insight generator for AttritionIQ.

Calculates executive KPIs, multi-dimensional attrition slices,
and automated statistical findings grounded directly in data.
"""

from typing import Dict, Any, List, Optional
import os
import pandas as pd
import numpy as np

from src.features.preprocessing import clean_pct_series


def load_and_prepare_historical_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Loads and sanitizes raw historical HR dataset with robust multi-location fallback."""
    if file_path is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates = [
            os.path.join(root_dir, "data", "raw", "HR_Employee_Data.csv"),
            os.path.join(root_dir, "HR_Employee_Data.csv"),
            os.path.join("data", "raw", "HR_Employee_Data.csv"),
            "HR_Employee_Data.csv"
        ]
        file_path = next((c for c in candidates if os.path.exists(c)), "data/raw/HR_Employee_Data.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Historical dataset not found at {file_path}")

    df = pd.read_csv(file_path)
    df["satisfaction_level"] = clean_pct_series(df["satisfaction_level"])
    df["last_evaluation"] = clean_pct_series(df["last_evaluation"])
    return df


def calculate_kpis(df: pd.DataFrame, test_recall: Optional[float] = None) -> Dict[str, Any]:
    """Computes headline workforce metrics."""
    total_employees = len(df)
    left_count = int((df["left"] == 1).sum()) if "left" in df.columns else 0
    stayed_count = total_employees - left_count
    attrition_rate = round((left_count / total_employees * 100.0), 2) if total_employees > 0 else 0.0

    avg_satisfaction = round(float(df["satisfaction_level"].mean() * 100.0), 1)
    avg_hours = round(float(df["average_montly_hours"].mean()), 1)
    avg_tenure = round(float(df["time_spend_company"].mean()), 1)

    return {
        "total_employees": total_employees,
        "left_count": left_count,
        "stayed_count": stayed_count,
        "attrition_rate_pct": attrition_rate,
        "avg_satisfaction_pct": avg_satisfaction,
        "avg_monthly_hours": avg_hours,
        "avg_tenure_years": avg_tenure,
        "model_recall_pct": round(test_recall * 100.0, 1) if test_recall is not None else None
    }


def get_department_attrition_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates attrition rates and counts grouped by department."""
    grouped = df.groupby("Department").agg(
        total_employees=("left", "count"),
        departures=("left", "sum"),
        avg_satisfaction=("satisfaction_level", "mean"),
        avg_hours=("average_montly_hours", "mean")
    ).reset_index()

    grouped["attrition_rate_pct"] = np.round(
        (grouped["departures"] / grouped["total_employees"]) * 100.0, 2
    )
    grouped["stayed_count"] = grouped["total_employees"] - grouped["departures"]
    return grouped.sort_values(by="attrition_rate_pct", ascending=False)


def get_salary_attrition_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates attrition statistics segmented by salary band."""
    salary_order = ["low", "medium", "high"]
    grouped = df.groupby("salary").agg(
        total_employees=("left", "count"),
        departures=("left", "sum"),
        avg_satisfaction=("satisfaction_level", "mean"),
        avg_hours=("average_montly_hours", "mean")
    ).reset_index()

    grouped["attrition_rate_pct"] = np.round(
        (grouped["departures"] / grouped["total_employees"]) * 100.0, 2
    )
    grouped["stayed_count"] = grouped["total_employees"] - grouped["departures"]
    # Sort logically
    grouped["salary"] = pd.Categorical(grouped["salary"], categories=salary_order, ordered=True)
    return grouped.sort_values("salary")


def get_tenure_attrition_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates attrition rates across years at company (tenure)."""
    grouped = df.groupby("time_spend_company").agg(
        total_employees=("left", "count"),
        departures=("left", "sum"),
        avg_satisfaction=("satisfaction_level", "mean")
    ).reset_index()

    grouped["attrition_rate_pct"] = np.round(
        (grouped["departures"] / grouped["total_employees"]) * 100.0, 2
    )
    grouped["stayed_count"] = grouped["total_employees"] - grouped["departures"]
    return grouped.sort_values(by="time_spend_company")


def get_hours_distribution_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates monthly hours into brackets and calculates attrition rate."""
    bins = [0, 150, 200, 250, 350]
    labels = ["Underworked (<150h)", "Balanced (150-200h)", "Demanding (200-250h)", "Overtime (>250h)"]
    df_temp = df.copy()
    df_temp["hours_bracket"] = pd.cut(df_temp["average_montly_hours"], bins=bins, labels=labels)

    grouped = df_temp.groupby("hours_bracket", observed=False).agg(
        total=("left", "count"),
        departures=("left", "sum")
    ).reset_index()
    grouped["attrition_rate_pct"] = np.round((grouped["departures"] / grouped["total"]) * 100.0, 2)
    return grouped


def get_satisfaction_distribution_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates satisfaction level brackets."""
    bins = [0.0, 0.30, 0.50, 0.70, 1.01]
    labels = ["Low (<30%)", "Moderate (30-50%)", "Satisfied (50-70%)", "High (>70%)"]
    df_temp = df.copy()
    df_temp["sat_bracket"] = pd.cut(df_temp["satisfaction_level"], bins=bins, labels=labels)

    grouped = df_temp.groupby("sat_bracket", observed=False).agg(
        total=("left", "count"),
        departures=("left", "sum")
    ).reset_index()
    grouped["attrition_rate_pct"] = np.round((grouped["departures"] / grouped["total"]) * 100.0, 2)
    return grouped


def generate_workforce_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Grounded automated statistical insights generated from historical employee data."""
    insights = []
    baseline_rate = (df["left"] == 1).mean() * 100.0

    # 1. Salary insight
    sal_stats = get_salary_attrition_stats(df)
    low_sal = sal_stats[sal_stats["salary"] == "low"].iloc[0]
    high_sal = sal_stats[sal_stats["salary"] == "high"].iloc[0]
    insights.append({
        "title": "Compensation Disparity Risk",
        "category": "Salary",
        "severity": "High",
        "text": (
            f"Employees on 'low' compensation observe an attrition rate of {low_sal['attrition_rate_pct']:.1f}%, "
            f"compared to only {high_sal['attrition_rate_pct']:.1f}% for 'high' compensation peers "
            f"({round(low_sal['attrition_rate_pct'] / max(high_sal['attrition_rate_pct'], 0.1), 1)}x difference). "
            f"This represents an elevated flight risk across early-career staff."
        )
    })

    # 2. Tenure friction insight
    tenure_stats = get_tenure_attrition_stats(df)
    peak_tenure = tenure_stats.loc[tenure_stats["attrition_rate_pct"].idxmax()]
    insights.append({
        "title": "Tenure Milestone Vulnerability",
        "category": "Tenure",
        "severity": "Medium",
        "text": (
            f"Attrition reaches a sharp peak at year {int(peak_tenure['time_spend_company'])} "
            f"with {peak_tenure['attrition_rate_pct']:.1f}% departure rate, vs the {baseline_rate:.1f}% historical baseline. "
            f"Employees at 3-5 years tenure represent the core organizational retention window."
        )
    })

    # 3. Department insight
    dept_stats = get_department_attrition_stats(df)
    top_dept = dept_stats.iloc[0]
    lowest_dept = dept_stats.iloc[-1]
    insights.append({
        "title": "Departmental Turnover Variance",
        "category": "Department",
        "severity": "Medium",
        "text": (
            f"The '{top_dept['Department']}' department demonstrates the highest historical departure rate "
            f"at {top_dept['attrition_rate_pct']:.1f}%, whereas '{lowest_dept['Department']}' shows lowest turnover "
            f"at {lowest_dept['attrition_rate_pct']:.1f}%. Cross-departmental workload audits are recommended."
        )
    })

    # 4. Overtime & Burnout
    overtime_df = df[df["average_montly_hours"] > 250]
    if len(overtime_df) > 0:
        ot_rate = (overtime_df["left"] == 1).mean() * 100.0
        insights.append({
            "title": "Workload Saturation & Burnout",
            "category": "Workload",
            "severity": "High",
            "text": (
                f"Employees logging >250 monthly hours exhibit an attrition rate of {ot_rate:.1f}%. "
                f"Sustained extreme hours correlate heavily with attrition, especially among senior contributors."
            )
        })

    return insights
