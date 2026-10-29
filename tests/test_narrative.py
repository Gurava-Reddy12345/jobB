"""Unit tests for the Plain-English HR Executive Narrative generator."""

import pytest
from unittest.mock import patch
from src.analytics.narrative import (
    format_promotion_tenure_phrase,
    generate_deterministic_plain_english_summary,
    generate_llm_plain_english_summary
)


def test_format_promotion_tenure_phrase_under_5_years():
    """Confirms logical consistency when tenure is less than 5 years."""
    # 3 years, no promotion
    phrase_no_promo = format_promotion_tenure_phrase(tenure=3, promoted=0)
    assert "entire tenure" in phrase_no_promo.lower()
    assert "last 5 years" not in phrase_no_promo.lower()
    assert "3 years" in phrase_no_promo

    # 2 years, with promotion
    phrase_promo = format_promotion_tenure_phrase(tenure=2, promoted=1)
    assert "during their tenure" in phrase_promo.lower()
    assert "last 5 years" not in phrase_promo.lower()
    assert "2 years" in phrase_promo


def test_format_promotion_tenure_phrase_5_years_or_more():
    """Confirms reference to 'last 5 years' when tenure is 5 or more years."""
    # 5 years, no promotion
    phrase_5yr = format_promotion_tenure_phrase(tenure=5, promoted=0)
    assert "last 5 years" in phrase_5yr.lower()
    assert "5 years" in phrase_5yr

    # 8 years, promoted
    phrase_8yr = format_promotion_tenure_phrase(tenure=8, promoted=1)
    assert "last 5 years" in phrase_8yr.lower()
    assert "8 years" in phrase_8yr


def test_deterministic_summary_structure_and_no_jargon():
    """Ensures deterministic narrative covers key dimensions without algorithmic jargon."""
    record = {
        "satisfaction_level": 0.35,
        "last_evaluation": 0.52,
        "number_project": 2,
        "average_montly_hours": 145,
        "time_spend_company": 3,
        "promotion_last_5years": 0,
        "Work_accident": 0,
        "Department": "support",
        "salary": "medium"
    }

    summary = generate_deterministic_plain_english_summary(
        record=record,
        probability=0.726,
        risk_category="High Risk"
    )

    # Validate content elements
    assert "72.6%" in summary
    assert "high risk" in summary.lower()
    assert "satisfaction" in summary.lower()
    assert "support department" in summary.lower() or "support" in summary.lower()
    assert "entire tenure" in summary.lower()

    # Strict check: No misleading 5-year phrase for 3-year tenure
    assert "promotion in the last 5 years" not in summary.lower()

    # Algorithmic jargon blacklist check
    jargon_blacklist = ["shap", "p-value", "hyperplane", "random forest", "gradient", "classifier", "overfitting"]
    for word in jargon_blacklist:
        assert word not in summary.lower(), f"Forbidden ML jargon '{word}' found in plain-English narrative"


def test_llm_summary_graceful_fallback():
    """Ensures LLM summary gracefully falls back to deterministic output when network fails."""
    record = {
        "satisfaction_level": 0.85,
        "last_evaluation": 0.90,
        "number_project": 4,
        "average_montly_hours": 200,
        "time_spend_company": 6,
        "promotion_last_5years": 1,
        "Work_accident": 0,
        "Department": "sales",
        "salary": "high"
    }

    with patch("requests.post", side_effect=Exception("Connection timed out")):
        summary = generate_llm_plain_english_summary(
            record=record,
            probability=0.12,
            risk_category="Low Risk",
            api_key="mock_key_for_testing"
        )

    assert isinstance(summary, str)
    assert len(summary) > 50
    assert "12.0%" in summary or "12%" in summary
    assert "low risk" in summary.lower()


def test_llm_summary_success_mock():
    """Ensures LLM response is returned when API returns valid 200 response."""
    record = {
        "satisfaction_level": 0.60,
        "last_evaluation": 0.70,
        "number_project": 3,
        "average_montly_hours": 180,
        "time_spend_company": 3,
        "promotion_last_5years": 0,
        "Work_accident": 0,
        "Department": "marketing",
        "salary": "medium"
    }

    class MockResponse:
        status_code = 200
        def json(self):
            return {
                "choices": [{
                    "message": {
                        "content": "This marketing professional demonstrates steady performance with moderate flight likelihood."
                    }
                }]
            }

    with patch("requests.post", return_value=MockResponse()):
        summary = generate_llm_plain_english_summary(
            record=record,
            probability=0.30,
            risk_category="Low Risk",
            api_key="mock_key"
        )

    assert "marketing professional" in summary
