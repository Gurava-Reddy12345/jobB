"""Plain-English HR Executive Narrative Generator for AttritionIQ.

Translates complex algorithmic predictions, feature interactions, and risk probabilities
into accessible, human-first plain English for non-technical HR managers and business leaders.
Supports OpenRouter LLM integration with automatic deterministic fallbacks.
"""

import os
import json
from typing import Dict, Any, Optional
import requests

# Model fallback chain on OpenRouter
DEFAULT_MODELS = [
    "nvidia/nemotron-3.5-lightning:free",
    "liquid/lfm-2.5-2.6b:free",
    "openrouter/auto"
]


def format_promotion_tenure_phrase(tenure: int, promoted: int) -> str:
    """Corrects logical mismatch where tenure < 5 years cannot logically refer to 'the last 5 years'.

    If tenure < 5:
        - 0 promotions: 'without receiving a promotion during their entire tenure'
        - 1 promotion: 'having received a promotion during their tenure'
    If tenure >= 5:
        - 0 promotions: 'without receiving a promotion in the last 5 years'
        - 1 promotion: 'having received a promotion within the past 5 years'
    """
    if tenure < 5:
        if promoted == 1:
            return f"They have been with the company for {tenure} years and have received a promotion during their tenure."
        else:
            return f"They have spent {tenure} years with the company without receiving a promotion during their entire tenure."
    else:
        if promoted == 1:
            return f"They have been with the company for {tenure} years, with at least one promotion recorded in the last 5 years."
        else:
            return f"They have been with the company for {tenure} years without a recorded promotion in the last 5 years."


def generate_deterministic_plain_english_summary(
    record: Dict[str, Any],
    probability: float,
    risk_category: str
) -> str:
    """Generates an accurate, professional, plain-English HR summary using domain rules.

    Guarantees logical correctness, zero algorithmic jargon, and empathetic HR context.
    """
    prob_pct = round(probability * 100.0, 1)
    sat = float(record.get("satisfaction_level", 0.5))
    eval_score = float(record.get("last_evaluation", 0.5))
    projects = int(record.get("number_project", 3))
    hours = int(record.get("average_montly_hours", 200))
    tenure = int(record.get("time_spend_company", 3))
    promoted = int(record.get("promotion_last_5years", 0))
    dept = str(record.get("Department", "General")).capitalize()
    salary = str(record.get("salary", "medium")).lower()
    accident = int(record.get("Work_accident", 0))

    sentences = []

    # 1. Headline Assessment
    sentences.append(
        f"The system estimates a {prob_pct}% likelihood that this employee may leave the company, "
        f"placing them in the {risk_category.lower()} tier."
    )

    # 2. Satisfaction Score
    if sat < 0.40:
        sentences.append(
            f"Their job satisfaction score is {sat:.2f} out of 1.0, which indicates substantial discontent "
            f"or dissatisfaction with their current daily experience."
        )
    elif sat < 0.65:
        sentences.append(
            f"Their job satisfaction score stands at {sat:.2f} out of 1.0, suggesting moderate contentment "
            f"with noticeable opportunities for morale and engagement improvements."
        )
    else:
        sentences.append(
            f"Their job satisfaction score is high at {sat:.2f} out of 1.0, representing a strong protective factor "
            f"that supports retention."
        )

    # 3. Project Load & Monthly Hours
    if hours > 240 and projects >= 5:
        sentences.append(
            f"They manage {projects} projects while logging approximately {hours} monthly hours (>55–60 hours/week). "
            f"This heavy workload represents a critical burnout hazard that warrants immediate task rebalancing."
        )
    elif hours < 150 and projects <= 2:
        sentences.append(
            f"They handle {projects} projects and log around {hours} hours per month. "
            f"This is notably below full capacity, which may signal disengagement, lack of challenge, or underutilization."
        )
    else:
        sentences.append(
            f"They currently manage {projects} concurrent projects with an average of {hours} working hours per month, "
            f"which reflects a balanced and manageable workload volume."
        )

    # 4. Tenure & Promotion (Strictly grammatically corrected for tenure < 5 vs >= 5)
    sentences.append(format_promotion_tenure_phrase(tenure, promoted))

    # 5. Performance Evaluation & Underappreciation Gap
    if eval_score >= 0.75 and sat < 0.45:
        sentences.append(
            f"Their latest performance evaluation was strong ({eval_score:.2f} out of 1.0), but their low satisfaction "
            f"creates an 'underappreciated performer' scenario where valuable contributors often look elsewhere for recognition."
        )
    elif eval_score >= 0.75:
        sentences.append(
            f"Their most recent performance rating was high ({eval_score:.2f} out of 1.0), demonstrating strong competence "
            f"and tangible value to their team."
        )
    else:
        sentences.append(
            f"Their recent performance evaluation is {eval_score:.2f} out of 1.0; HR and their manager should review "
            f"whether additional mentorship, clearer expectations, or skill development could help them thrive."
        )

    # 6. Safety & Work Accident
    if accident == 1:
        sentences.append(
            "They have a past workplace accident on record, so HR should ensure they feel fully supported and safe in their environment."
        )
    else:
        sentences.append(
            "They have no recorded workplace accidents, which confirms physical safety is not a primary factor influencing their standing."
        )

    # 7. Department & Compensation Context
    sentences.append(
        f"They work in the {dept} department and receive a '{salary}' salary-band compensation. "
        f"HR should verify whether their compensation and benefits align competitively with peer roles inside and outside the company."
    )

    # 8. Human-in-the-Loop Actionable Closing
    sentences.append(
        "Taken together, these observations suggest that a constructive, empathetic one-on-one conversation "
        "focusing on daily satisfaction, career aspirations, and workload support would be highly beneficial. "
        "These points are exploratory guideposts to assist leaders—only a direct personal dialogue can reveal how the employee truly feels."
    )

    return " ".join(sentences)


def _resolve_api_key(provided_key: Optional[str] = None) -> Optional[str]:
    """Resolves the OpenRouter API key from passed argument, environment variables,
    Streamlit secrets, or local configuration files without hardcoding keys in source control.
    """
    if provided_key and not provided_key.startswith("your_"):
        return provided_key

    # 1. Environment variable
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key and not env_key.startswith("your_"):
        return env_key

    # 2. Streamlit secrets (when running inside Streamlit Cloud / app)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENROUTER_API_KEY" in st.secrets:
            sec_key = st.secrets["OPENROUTER_API_KEY"]
            if sec_key and not sec_key.startswith("your_"):
                return sec_key
    except Exception:
        pass

    # 3. Local .streamlit/secrets.toml or .env fallback file read
    try:
        from pathlib import Path
        base_dir = Path(__file__).resolve().parent.parent.parent
        secrets_file = base_dir / ".streamlit" / "secrets.toml"
        if secrets_file.exists():
            with open(secrets_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "OPENROUTER_API_KEY" in line and "=" in line:
                        k = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if k and not k.startswith("your_"):
                            return k

        env_file = base_dir / ".env"
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("OPENROUTER_API_KEY=") and "=" in line:
                        k = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if k and not k.startswith("your_"):
                            return k
    except Exception:
        pass

    return None


def generate_llm_plain_english_summary(
    record: Dict[str, Any],
    probability: float,
    risk_category: str,
    api_key: Optional[str] = None,
    preferred_model: str = "nvidia/nemotron-3.5-lightning:free"
) -> str:
    """Generates an executive plain-English summary using OpenRouter API with automated fallbacks.

    Falls back to deterministic rule generation if network or API keys are unavailable.
    """
    resolved_key = _resolve_api_key(api_key)

    if not resolved_key:
        return generate_deterministic_plain_english_summary(record, probability, risk_category)

    # Construct clean prompt with domain constraints
    tenure = int(record.get("time_spend_company", 3))
    promoted = int(record.get("promotion_last_5years", 0))
    correct_tenure_phrase = format_promotion_tenure_phrase(tenure, promoted)

    prompt = f"""
You are an empathetic, highly experienced Human Resources Executive writing an executive employee retention summary for a non-technical manager.

Translate the following employee profile and flight risk assessment into a cohesive, warm, and professional plain-English narrative (around 150-250 words).

Key Rules:
1. DO NOT use technical machine learning jargon (no mentions of 'SHAP', 'p-values', 'trees', 'classifiers', or 'algorithms').
2. Explain the estimated probability and risk tier in conversational business language.
3. Crucial grammar rule for tenure: The employee has been with the company for {tenure} years. {correct_tenure_phrase}. Do NOT state 'without a promotion in the last 5 years' if tenure is less than 5 years!
4. Connect workload (monthly hours and project count) and satisfaction to burnout and morale.
5. Provide actionable, compassionate suggestions for how their manager and HR can support them in a positive 1-on-1 discussion.
6. Emphasize that these are helpful conversational guideposts, not certain proof that the person will leave.

Employee Data:
- Estimated Flight Risk: {round(probability * 100.0, 1)}% ({risk_category})
- Satisfaction Score: {record.get('satisfaction_level')} (scale 0.0 to 1.0)
- Last Performance Evaluation: {record.get('last_evaluation')} (scale 0.0 to 1.0)
- Number of Projects: {record.get('number_project')}
- Average Monthly Hours: {record.get('average_montly_hours')} hours/month
- Company Tenure: {tenure} years
- Promoted in last 5 years: {'Yes' if promoted == 1 else 'No'}
- Department: {record.get('Department')}
- Salary Band: {record.get('salary')}
- Workplace Accident on Record: {'Yes' if record.get('Work_accident') == 1 else 'No'}

Write the narrative now in flowing, natural paragraphs:
"""

    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Svamsi2006/jobB",
        "X-Title": "AttritionIQ Workforce Platform"
    }

    # Try preferred model first, then candidate fallback models
    models_to_try = [preferred_model] + [m for m in DEFAULT_MODELS if m != preferred_model]

    for model in models_to_try:
        try:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an empathetic HR director writing clear, non-technical plain English workforce summaries."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 450
            }
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=12
            )
            if resp.status_code == 200:
                result = resp.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if content:
                    return content
        except Exception:
            continue

    # Fallback to deterministic generator if API fails or times out
    return generate_deterministic_plain_english_summary(record, probability, risk_category)
