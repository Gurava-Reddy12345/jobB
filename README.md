# AttritionIQ: Explainable Employee Attrition Prediction & Workforce Analytics Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-00E5FF.svg)](https://shap.readthedocs.io/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> **AttritionIQ** is an enterprise-grade, explainable machine learning platform and workforce intelligence dashboard designed to predict employee flight risk, uncover macro retention patterns, simulate policy interventions via what-if counterfactuals, and provide transparent model reliability audits.

---

## 🌟 Key Capabilities

1. **Predictive Machine Learning**: Accurately estimates employee departure probability with **97.3% Test F1-Score** and **0.990 Test ROC-AUC** across held-out employee cohorts.
2. **Explainable AI (SHAP)**: Employs game-theoretic `TreeExplainer` attribution to reveal both flight-risk triggers and protective factors for every single prediction.
3. **Macro Workforce Intelligence**: Interactive slicing across departments, salary tiers, and tenure milestones, supplemented by automatically generated empirical statistical insights.
4. **What-If Scenario Simulator**: Simulates counterfactual policy interventions (e.g., workload rebalancing, compensation shifts) and compares predicted flight risk shifts side-by-side.
5. **Data Quality & Model Governance**: Audits duplicate feature profiles, isolates historical targets to eliminate data leakage, benchmarks group-based vs. random-split generalization, and generates downloadable audit reports.
6. **Batch Inference Engine**: Uploads, validates, cleans, and scores entire workforce CSV cohorts with real-time row-by-row error diagnostics and CSV report export.
7. **Plain-English Executive HR Narratives**: Generates empathetic, non-technical executive summaries powered by OpenRouter LLM (`nvidia/nemotron-3.5-lightning:free`) with automated deterministic fallback and rigorous tenure-grammar validation (preventing logical contradictions for employees with `<5` years tenure).

---

## 🏗️ Project Architecture & File Hierarchy

```text
AttritionIQ/
│
├── app.py                          # Executive platform landing hub & navigation
├── requirements.txt                # Pinned production dependencies
├── README.md                       # Comprehensive project documentation
├── ATTRITION_IQ_DEEP_DIVE.md       # Detailed technical methodology & parameter breakdown
├── .gitignore                      # Git exclusion rules
├── .env.example                    # Environment configuration template
│
├── data/
│   ├── raw/
│   │   └── HR_Employee_Data.csv    # Source of truth historical employee dataset (14,999 records)
│   ├── processed/
│   │   └── cleaned_features.csv    # Sanitized, engineered feature dataset
│   └── sample/
│       └── sample_new_employees.csv# Batch prediction upload template (synthetic)
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Exploratory data analysis & prototyping notebook
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── validation.py           # Schema enforcement, range checks, target leak prevention
│   ├── features/
│   │   ├── __init__.py
│   │   └── preprocessing.py        # Percentage cleaning, categorical encoding, feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                # Model training, comparison, group-split audit & serialization
│   │   ├── evaluate.py             # Metrics calculation (F1, ROC, PR, calibration, Brier score)
│   │   ├── predict.py              # Single & batch inference engine, risk tiers, HR recommendations
│   │   └── explain.py              # SHAP individual attributions & waterfall generation
│   └── analytics/
│       ├── __init__.py
│       ├── workforce.py            # Workforce KPIs, multi-dimensional slices & automated insights
│       └── narrative.py            # Plain-English HR narrative generator (OpenRouter LLM & fallback)
│
├── pages/
│   ├── 1_Workforce_Overview.py     # Screen 1: Macro workforce analytics & automated insights
│   ├── 2_Employee_Risk_Analysis.py # Screen 2: Individual risk scoring, SHAP waterfall & executive narrative
│   ├── 3_Batch_Predictions.py      # Screen 3: CSV cohort upload, validation & batch scoring
│   ├── 4_Model_Performance.py      # Screen 4: Model benchmarks, ROC/PR curves & reliability panel
│   ├── 5_What_If_Lab.py            # Screen 5: Counterfactual policy simulator
│   └── 6_Data_Quality_Audit.py     # Screen 6: Data hygiene, duplicate profile audit & lineage
│
├── components/
│   ├── __init__.py
│   ├── charts.py                   # Reusable Plotly visualizations
│   ├── metrics.py                  # Styled KPI cards, risk badges, and alert banners
│   └── sidebar.py                  # Executive dark-navy sidebar navigation
│
├── artifacts/
│   ├── model_pipeline.joblib       # Serialized champion pipeline (Preprocessor + Fitted Model)
│   └── model_metadata.json         # Model version, split sizes, and top 10 feature importances
│
├── reports/
│   ├── model_evaluation.json       # Complete test metrics, ROC points, PR curve & group-split audit
│   └── data_quality_report.json    # Complete data hygiene and schema verification report
│
└── tests/
    ├── __init__.py
    ├── test_data_validation.py     # Unit tests for validation, range bounds & aliases
    ├── test_preprocessing.py       # Unit tests for feature engineering & column transformers
    ├── test_predictions.py         # Unit tests for prediction logic, thresholds & risk categories
    └── test_narrative.py           # Unit tests for plain-English narrative & grammar consistency
```

---

## ⚡ Engineered Parameters: Making the Model Sharp

When inspecting the raw historical dataset (`HR_Employee_Data.csv`), standard raw predictors missed critical cross-feature behavioral dynamics (such as burnout and underappreciation). To maximize predictive precision and recall, we engineered 10 domain-specific parameters:

| Engineered Feature | Formulation | Behavioral Signal & Rationale | Feature Importance Rank |
|---|---|---|:---:|
| `eval_sat_diff` | `last_evaluation - satisfaction_level` | **Underappreciated Performer Gap**: Star performers with low satisfaction are prime flight risks. | **Top 5 (8.34%)** |
| `projects_per_year` | `number_project / time_spend_company` | **Project Velocity**: Workload intensity relative to tenure. | **Top 4 (8.75%)** |
| `hours_per_project` | `average_montly_hours / number_project` | **Workload Concentration**: Differentiates multi-tasking overload from deep-focus allocation. | **Top 8 (5.37%)** |
| `overworked` | `hours > 240 and projects >= 5` | **Overwork / Overtime Flag**: Identifies chronic overtime contributors. | **Top 9 (5.16%)** |
| `burnout_risk` | `eval > 0.75 and hours > 220 and sat < 0.45` | **High-Performer Burnout Indicator**: Identifies critical talent at immediate risk of exhaustion. | **Significant** |
| `tenure_risk_window` | `tenure in [3, 4, 5]` | **Tenure Friction Milestone**: Captures empirical turnover surge observed at years 3 to 5. | **Significant** |
| `unrewarded_high_performer` | `eval > 0.8 and promo == 0 and salary != high` | **Career Plateau**: Identifies unpromoted top talent vulnerable to external headhunting. | **Significant** |
| `salary_num` | `low: 1, medium: 2, high: 3` | Quantifies compensation tier for numerical computation. | **Top 10** |
| `salary_workload_ratio` | `salary_num / (hours / 100)` | **Effort-to-Compensation Equity**: Measures whether hours worked match compensation level. | **Top 10 (3.43%)** |
| `underworked` | `hours < 150 and projects <= 2` | **Disengagement Signal**: Identifies underutilized employees drifting toward attrition. | **Significant** |

### Mathematical Formulations (GitHub Block Math)

$$
\text{eval\_sat\_diff} = \text{last\_evaluation} - \text{satisfaction\_level}
$$

$$
\text{projects\_per\_year} = \frac{\text{number\_project}}{\max(1, \text{time\_spend\_company})}
$$

$$
\text{hours\_per\_project} = \frac{\text{average\_montly\_hours}}{\max(1, \text{number\_project})}
$$

$$
\text{burnout\_risk} = \mathbb{I}(\text{last\_evaluation} > 0.75 \land \text{average\_montly\_hours} > 220 \land \text{satisfaction\_level} < 0.45)
$$

$$
\text{overworked} = \mathbb{I}(\text{average\_montly\_hours} > 240 \land \text{number\_project} \ge 5)
$$

$$
\text{salary\_workload\_ratio} = \frac{\text{salary\_num}}{\text{average\_montly\_hours} / 100}
$$

---

## 📊 Model Evaluation Benchmark (Held-Out Test Set)

All metrics were evaluated on **2,250 strictly held-out test records** (15% test split, stratified by target):

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Brier Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Champion)** | **98.80%** | **99.04%** | **95.71%** | **0.9734** | **0.9901** | **0.9859** | **0.0102** |
| **HistGradientBoosting** | 98.31% | 96.79% | 95.90% | 0.9634 | 0.9902 | 0.9818 | 0.0135 |
| **Logistic Regression (Baseline)** | 87.20% | 67.17% | 90.49% | 0.7711 | 0.9365 | 0.8346 | 0.0972 |

### 🛡️ Group-Based Split vs Random-Split Diagnostic

The historical dataset contains **3,008 duplicate feature profiles** (records with identical feature values excluding `Emp_Id`). To verify that our Random Forest model was not simply memorizing repeated rows:
- **Standard Random-Split F1**: `0.9734`
- **Group-Based Split F1** (where all identical profiles are grouped into either train or test): `0.9650`
- **Generalization Delta**: `-0.0084` (< 1.5% drop)

This confirms the model has captured genuine underlying workforce dynamics rather than trivial row memorization.

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/Svamsi2006/jobB.git
cd jobB

# Create virtual environment (optional but recommended)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Automated Test Suite
Verify that validation rules, feature engineering, and inference pipelines pass all unit tests:

```bash
python -m pytest tests/
```

### 3. Train & Evaluate Models
Run the end-to-end training pipeline to benchmark candidate models, compute SHAP explainers, and generate JSON evaluation reports:

```bash
python -m src.models.train
```

### 4. Launch the AttritionIQ Platform
Start the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The web application will open automatically at `http://localhost:8501`.

---

## 🖥️ Screen Walkthrough

| Screen | File | Highlights |
|---|---|---|
| **Executive Hub** | `app.py` | SaaS hero banner, headline KPIs (14,999 records, 23.81% attrition, 95.7% recall), module launchpad. |
| **1. Workforce Overview** | `pages/1_Workforce_Overview.py` | Interactive filters (department, salary, tenure), 6 Plotly charts, dynamic automated workforce insights. |
| **2. Employee Risk Analysis** | `pages/2_Employee_Risk_Analysis.py` | Individual input form, instant preset scenario buttons, flight risk gauge, SHAP waterfall chart, recommended actions. |
| **3. Batch Predictions** | `pages/3_Batch_Predictions.py` | CSV cohort uploader, template download, row-level error validation, cohort risk distributions, enriched CSV export. |
| **4. Model Performance** | `pages/4_Model_Performance.py` | 3-model benchmark table, interactive Confusion Matrix, ROC curves, PR curves, calibration curves, group-split audit. |
| **5. What-If Scenario Lab** | `pages/5_What_If_Lab.py` | Baseline vs. alternative intervention simulator, net risk delta KPI, comparison matrix, causal disclaimer. |
| **6. Data Quality & Audit** | `pages/6_Data_Quality_Audit.py` | Data hygiene audit, duplicate profile analysis, target isolation verification, JSON audit report download. |

---

## ⚖️ Ethical Considerations & Methodological Limitations

1. **Predictive Associations, Not Causality**: Predicted flight probabilities and SHAP attributions reflect statistical correlations within historical data. They do not prove that altering an attribute guarantees an employee will stay.
2. **Absence of Timestamps**: The dataset does not include temporal timestamps. Consequently, prospective time-series forecasting cannot be verified from this CSV alone.
3. **Supportive Talent Management**: Predictions are intended as diagnostic signals to guide proactive management check-ins, workload reviews, and supportive 1-on-1s—never as automated grounds for disciplinary or severance decisions.

---

## 📄 License
This project is licensed under the MIT License.
