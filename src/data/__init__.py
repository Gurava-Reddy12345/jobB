"""Data ingestion and validation module."""
from .validation import validate_employee_data, validate_single_record, get_data_quality_summary

__all__ = ["validate_employee_data", "validate_single_record", "get_data_quality_summary"]
