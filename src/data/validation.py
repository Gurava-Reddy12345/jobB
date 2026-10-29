"""Data validation and quality audit for AttritionIQ.

Enforces schema constraints, domain ranges, checks duplicate profiles,
and detects data hygiene issues in training and inference records.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

REQUIRED_PREDICTORS = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "Department",
    "salary"
]

ALLOWED_DEPARTMENTS = [
    "sales",
    "technical",
    "support",
    "IT",
    "product_mng",
    "marketing",
    "RandD",
    "accounting",
    "hr",
    "management"
]

ALLOWED_SALARIES = ["low", "medium", "high"]

COLUMN_ALIASES = {
    "average_monthly_hours": "average_montly_hours",
    "department": "Department",
    "work_accident": "Work_accident",
    "time_spent_company": "time_spend_company",
    "time_spent": "time_spend_company",
    "emp_id": "Emp_Id",
    "id": "Emp_Id"
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names and maps known aliases."""
    renamed = df.copy()
    rename_map = {}
    for col in renamed.columns:
        clean_col = col.strip()
        lower_col = clean_col.lower()
        if clean_col in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[clean_col]
        elif lower_col in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[lower_col]
        elif clean_col.lower() == "department":
            rename_map[col] = "Department"
    if rename_map:
        renamed.rename(columns=rename_map, inplace=True)
    return renamed


def clean_percentage_value(val: Any) -> Optional[float]:
    """Converts percentage string (e.g., '38%') or float to standard decimal [0.0, 1.0]."""
    if pd.isna(val):
        return None
    if isinstance(val, str):
        val_clean = val.strip()
        has_pct = "%" in val_clean
        val_clean = val_clean.replace("%", "")
        try:
            num = float(val_clean)
            if has_pct:
                return num / 100.0
            if 0.0 <= num <= 1.0:
                return num
            elif 5.0 <= num <= 100.0 and num == int(num):
                return num / 100.0
            return num
        except ValueError:
            return None
    if isinstance(val, (int, float)):
        num = float(val)
        if 0.0 <= num <= 1.0:
            return num
        elif 5.0 <= num <= 100.0 and num == int(num):
            return num / 100.0
        return num
    return None



def validate_single_record(record: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validates a single employee profile dictionary before inference.

    Returns:
        (is_valid, error_messages, sanitized_record)
    """
    errors: List[str] = []
    sanitized: Dict[str, Any] = {}

    # Check for required fields
    for field in REQUIRED_PREDICTORS:
        if field not in record and field.lower() not in [k.lower() for k in record.keys()]:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return False, errors, record

    # Map keys
    input_data = {}
    for k, v in record.items():
        clean_k = COLUMN_ALIASES.get(k.strip().lower(), k.strip())
        if clean_k.lower() == "department":
            clean_k = "Department"
        input_data[clean_k] = v

    # satisfaction_level
    sat = clean_percentage_value(input_data.get("satisfaction_level"))
    if sat is None or not (0.0 <= sat <= 1.0):
        errors.append("satisfaction_level must be between 0.0 and 1.0 (or 0% - 100%)")
    else:
        sanitized["satisfaction_level"] = float(sat)

    # last_evaluation
    eval_val = clean_percentage_value(input_data.get("last_evaluation"))
    if eval_val is None or not (0.0 <= eval_val <= 1.0):
        errors.append("last_evaluation must be between 0.0 and 1.0 (or 0% - 100%)")
    else:
        sanitized["last_evaluation"] = float(eval_val)

    # number_project
    try:
        proj = int(input_data.get("number_project"))
        if not (1 <= proj <= 15):
            errors.append("number_project must be an integer between 1 and 15")
        else:
            sanitized["number_project"] = proj
    except (ValueError, TypeError):
        errors.append("number_project must be a valid integer")

    # average_montly_hours
    try:
        hours = int(input_data.get("average_montly_hours"))
        if not (40 <= hours <= 400):
            errors.append("average_montly_hours must be between 40 and 400")
        else:
            sanitized["average_montly_hours"] = hours
    except (ValueError, TypeError):
        errors.append("average_montly_hours must be a valid integer")

    # time_spend_company
    try:
        tenure = int(input_data.get("time_spend_company"))
        if not (1 <= tenure <= 30):
            errors.append("time_spend_company must be between 1 and 30 years")
        else:
            sanitized["time_spend_company"] = tenure
    except (ValueError, TypeError):
        errors.append("time_spend_company must be a valid integer")

    # Work_accident
    try:
        accident = int(input_data.get("Work_accident"))
        if accident not in (0, 1):
            errors.append("Work_accident must be 0 (No) or 1 (Yes)")
        else:
            sanitized["Work_accident"] = accident
    except (ValueError, TypeError):
        errors.append("Work_accident must be binary 0 or 1")

    # promotion_last_5years
    try:
        prom = int(input_data.get("promotion_last_5years"))
        if prom not in (0, 1):
            errors.append("promotion_last_5years must be 0 (No) or 1 (Yes)")
        else:
            sanitized["promotion_last_5years"] = prom
    except (ValueError, TypeError):
        errors.append("promotion_last_5years must be binary 0 or 1")

    # Department
    dept = str(input_data.get("Department", "")).strip()
    matched_dept = None
    for allowed in ALLOWED_DEPARTMENTS:
        if dept.lower() == allowed.lower():
            matched_dept = allowed
            break
    if not matched_dept:
        errors.append(f"Department '{dept}' is unrecognized. Must be one of: {', '.join(ALLOWED_DEPARTMENTS)}")
    else:
        sanitized["Department"] = matched_dept

    # salary
    sal = str(input_data.get("salary", "")).strip().lower()
    if sal not in ALLOWED_SALARIES:
        errors.append(f"salary '{sal}' must be one of: low, medium, high")
    else:
        sanitized["salary"] = sal

    if "Emp_Id" in input_data:
        sanitized["Emp_Id"] = str(input_data["Emp_Id"])

    return len(errors) == 0, errors, sanitized


def validate_employee_data(
    df: pd.DataFrame,
    is_training: bool = False
) -> Dict[str, Any]:
    """Validates a batch dataframe of employees.

    Args:
        df: Input DataFrame.
        is_training: If True, validates 'left' target column as well.

    Returns:
        Dictionary with validation status, clean dataframe, invalid records, and diagnostic stats.
    """
    normalized_df = normalize_column_names(df)
    errors: List[str] = []
    warnings: List[str] = []

    # Check for target leakage in inference mode
    target_stripped = False
    if not is_training and "left" in normalized_df.columns:
        warnings.append(
            "Target column 'left' was present in inference dataset. "
            "It has been isolated to prevent target leakage into model inference."
        )
        target_stripped = True

    # Check required columns
    missing_cols = [c for c in REQUIRED_PREDICTORS if c not in normalized_df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "valid_df": pd.DataFrame(),
            "invalid_df": df,
            "invalid_reasons": ["Missing required columns"] * len(df)
        }

    if is_training and "left" not in normalized_df.columns:
        errors.append("Missing target column 'left' in training dataset.")
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "valid_df": pd.DataFrame(),
            "invalid_df": df,
            "invalid_reasons": ["Missing target column 'left'"] * len(df)
        }

    valid_records = []
    invalid_records = []
    invalid_reasons = []

    for idx, row in normalized_df.iterrows():
        rec_dict = row.to_dict()
        is_rec_valid, rec_errors, sanitized = validate_single_record(rec_dict)

        if is_training:
            try:
                target_val = int(row["left"])
                if target_val not in (0, 1):
                    rec_errors.append(f"Invalid target value: {target_val}")
                    is_rec_valid = False
                else:
                    sanitized["left"] = target_val
            except Exception:
                rec_errors.append("Invalid non-integer target in 'left'")
                is_rec_valid = False

        if is_rec_valid:
            valid_records.append(sanitized)
        else:
            invalid_records.append(row.to_dict())
            invalid_reasons.append("; ".join(rec_errors))

    valid_df = pd.DataFrame(valid_records)
    invalid_df = pd.DataFrame(invalid_records)

    # Check duplicate employee profiles (excluding Emp_Id)
    duplicate_profiles_count = 0
    if not valid_df.empty:
        check_cols = [c for c in REQUIRED_PREDICTORS if c in valid_df.columns]
        duplicate_profiles_count = int(valid_df.duplicated(subset=check_cols).sum())

    return {
        "is_valid": len(valid_records) > 0,
        "total_records": len(df),
        "valid_count": len(valid_records),
        "invalid_count": len(invalid_records),
        "errors": errors,
        "warnings": warnings,
        "valid_df": valid_df,
        "invalid_df": invalid_df,
        "invalid_reasons": invalid_reasons,
        "duplicate_profiles_count": duplicate_profiles_count,
        "target_stripped": target_stripped
    }


def get_data_quality_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates comprehensive audit metrics for the dataset."""
    normalized = normalize_column_names(df)
    total_records = len(normalized)
    total_columns = len(normalized.columns)
    missing_counts = normalized.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())

    # Exact duplicates
    exact_duplicates = int(normalized.duplicated().sum())

    # Feature-only duplicates (excluding Emp_Id)
    feature_cols = [c for c in normalized.columns if c != "Emp_Id"]
    feature_duplicates = int(normalized.duplicated(subset=feature_cols).sum()) if feature_cols else 0

    # Target stats if available
    target_stats = {}
    if "left" in normalized.columns:
        left_counts = normalized["left"].value_counts().to_dict()
        stayed = int(left_counts.get(0, 0))
        left = int(left_counts.get(1, 0))
        rate = round((left / total_records) * 100.0, 2) if total_records > 0 else 0.0
        target_stats = {
            "stayed_count": stayed,
            "left_count": left,
            "attrition_rate_pct": rate
        }

    return {
        "total_records": total_records,
        "total_columns": total_columns,
        "columns_list": list(normalized.columns),
        "missing_counts": missing_counts,
        "total_missing_values": total_missing,
        "exact_duplicates": exact_duplicates,
        "repeated_feature_profiles": feature_duplicates,
        "target_stats": target_stats,
        "excluded_features": ["Emp_Id"] if "Emp_Id" in normalized.columns else [],
        "used_features": [c for c in REQUIRED_PREDICTORS if c in normalized.columns]
    }
