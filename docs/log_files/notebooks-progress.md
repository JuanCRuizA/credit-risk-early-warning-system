# Credit Risk Early Warning System (CREWS) - Notebooks Progress Log

## 01_eda.ipynb

**Date:** 2026-01-25 (updated 2026-02-15)
**Status:** Completed (61 cells)
**Location:** `notebooks/01_eda.ipynb`
**Objective:** Explore the Home Credit dataset structure, analyze the target variable, identify data quality issues, and document findings for the feature engineering phase.

### Datasets Loaded
- `application_train.csv`: 307,511 rows x 122 columns (main dataset)
- `bureau.csv`: 1,716,428 rows x 17 columns (credit bureau records)
- `bureau_balance.csv`: 27,299,925 rows x 3 columns (bureau monthly balances)
- 6 additional CSV files via automated loop (credit_card_balance, installments_payments, POS_CASH_balance, previous_application, HomeCredit_columns_description, sample_submission)

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Setup and Data Loading | Imports, paths, load application_train.csv |
| 2. Initial Data Exploration | Data types (106 numerical, 16 categorical), basic statistics |
| 2.1 EDA Application train | Pivot tables for all 16 categorical variables vs default rate |
| 2.2 EDA bureau | Bureau categorical analysis (CREDIT_ACTIVE, CREDIT_CURRENCY, CREDIT_TYPE) |
| 2.3 EDA bureau_balance | STATUS distribution analysis |
| 2.4 EDA remaining CSVs | Automated loop processing 6 additional files |
| 3. Target Variable Analysis | 8.07% default rate, bar chart + pie chart, imbalance warning |
| 4. Missing Values Analysis | 67 of 122 columns have missing data, horizontal bar chart |
| 4.1 Missing Value Patterns | Missing matrix heatmap, missingness correlation analysis |
| 4.2 Null Handling Strategy | Formal strategy by missing percentage threshold (DROP/KEEP/Impute) |
| 5. Key Feature Analysis | Statistics for 9 key features, distributions by target |
| 5.1 Correlation Analysis | Correlation heatmap, multicollinearity check (pairs with r > 0.7) |
| 5.2 Outlier Detection | Box plots for 4 financial features, IQR-based quantification |
| 6. External Source Scores | EXT_SOURCE correlation with target, decile analysis |
| 7. Categorical Features | Default rates by 6 key categorical features |
| 8. Summary and Next Steps | Consolidated findings, next steps roadmap |

### Visualizations (10 total, saved to `reports/`)
1. `target_distribution.png` - Bar chart + pie chart
2. `missing_values.png` - Top 30 columns with missing data
3. `missing_patterns.png` - Missing value matrix heatmap
4. `missing_correlation.png` - Missingness correlation between features
5. `correlation_heatmap.png` - Key numerical features correlation matrix
6. `outlier_boxplots.png` - Box plots for financial features by target
7. `feature_distributions.png` - 6 feature distributions by target
8. `ext_source_analysis.png` - Default rate by EXT_SOURCE_2 deciles
9. `categorical_analysis.png` - Default rates by 6 categorical features
10. `feature_correlations.png` - Top 20 correlations with target

### Key Findings
- **8.07% default rate** (282,686 non-defaults vs 24,825 defaults)
- **67 columns** have missing values; housing-related features share missing patterns
- **EXT_SOURCE scores** are the most predictive features (r = -0.15 to -0.18 with target)
- **DAYS_EMPLOYED = 365243** anomaly affects 18% of records (unemployed/retired)
- **Outliers** in AMT_INCOME_TOTAL (max $117M), documented but preserved for XGBoost
- **AMT_CREDIT and AMT_GOODS_PRICE** highly correlated (r = 0.99)
- **Occupation, income type, education** significantly impact default rates

### Key Variables Preserved for Downstream
- `df`, `dfb`, `dfbl`, `dataframes` dictionary
- `AGE_YEARS`, `EMPLOYMENT_YEARS`, `EXT_SOURCE_2_BIN` columns
- `missing_df`, `corr_matrix`, `categorical_cols`

---

## 02_FeatureEng.ipynb

**Date:** 2026-01-25
**Status:** Completed (36 cells)
**Location:** `notebooks/02_FeatureEng.ipynb`
**Objective:** Transform raw data into 200+ engineered features integrating application, bureau, and previous application data.

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Setup and Data Loading | Load application_train.csv (307,511 x 122) |
| 2. Application Features | 20 engineered features (ratios, time-based, external scores, documents) |
| 3. Bureau Features | 41 features from credit bureau (loan counts, credit types, ratios) |
| 4. Previous Application Features | 30 features from previous applications (status, contracts, ratios) |
| 5. Merge All Features | Left joins: application + bureau + previous applications |
| 6. Feature Selection | Final dataset preparation, missing value handling |
| 7. Summary | Feature count, save to CSV |

### Features Engineered (91 new features)

**From Application Train (20 features):**
- Ratio Features (6): DEBT_TO_INCOME, PAYMENT_BURDEN, CREDIT_TO_GOODS, ANNUITY_TO_CREDIT, INCOME_PER_PERSON, INCOME_TO_CREDIT
- Time-Based Features (6): AGE_YEARS, EMPLOYMENT_YEARS, FLAG_UNEMPLOYED, REGISTRATION_YEARS, ID_PUBLISH_YEARS, EMPLOYMENT_TO_AGE
- External Source Features (6): EXT_SOURCE_MEAN, EXT_SOURCE_WEIGHTED, EXT_SOURCE_PRODUCT, EXT_SOURCE_MIN, EXT_SOURCE_MAX, EXT_SOURCE_MISSING_COUNT
- Document Features (2): DOCUMENTS_PROVIDED_COUNT, DOCUMENTS_PROVIDED_RATIO

**From Bureau Data (41 features):**
- Loan Count Aggregations (22): BUREAU_LOAN_COUNT, BUREAU_ACTIVE_LOAN_COUNT, BUREAU_CLOSED_LOAN_COUNT, and various status counts
- Credit Type Counts (15): Counts by credit type (consumer, credit card, car loan, mortgage, etc.)
- Derived Ratios (4): BUREAU_DEBT_TO_CREDIT_RATIO, BUREAU_OVERDUE_RATIO, BUREAU_ACTIVE_RATIO, and others

**From Previous Applications (30 features):**
- Application Aggregations (18): Counts, approval/refusal rates, amounts
- Contract Status Features (6): By status (approved, refused, canceled, unused offer)
- Contract Type Features (4): By type (cash, consumer, revolving)
- Derived Features (2): Credit-to-application ratio, average credit per app

### Key Outputs
- `data/processed/features_train.csv` - 307,511 rows x 213 columns (363.2 MB)
- `data/processed/feature_names.csv` - List of 211 feature names
- `reports/feature_correlations.png` - Top 20 features correlated with target

### Key Findings
- Final feature count: **213 columns** (122 original + 91 engineered)
- **99.4%** of applicants have bureau data (305,811 / 307,511)
- External source scores remain the most predictive features across all transformations
- Bureau features filled with 0 for applicants without credit history
- All 307,511 observations preserved through merges

---

## 03_model_training_evaluation.ipynb

**Date:** 2026-01-25 (updated 2026-02-01)
**Status:** Completed (37 cells)
**Location:** `notebooks/03_model_training_evaluation.ipynb`
**Objective:** Train an XGBoost credit risk model with business cost optimization and production-ready threshold selection.

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Setup and Data Loading | Load features_train.csv (307,511 x 213) |
| 2. Data Preparation | Label encoding for 16 categorical columns, stratified train/test split |
| 3. Train XGBoost Model | 200 estimators, scale_pos_weight=11.39 |
| 4. Cross-Validation | 5-Fold stratified CV: Mean AUC 0.7757 (+/-0.0037) |
| 5. Threshold Optimization | Youden's J (0.5092) and business optimal (0.59) |
| 6. Business Cost Analysis | Profit maximization: $36.8M at threshold 0.59 |
| 7. Feature Importance | Top 25 features + category breakdown |
| 8. Model Calibration | Calibration curve for probability reliability |
| 9. Save Final Model | Pickle files for model, encoders, thresholds, features |
| 10. Summary | Key metrics, portfolio predictions, database storage |

### Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.7793 |
| Gini Coefficient | 0.5585 |
| KS Statistic | 0.421 |
| Cross-Validation AUC | 0.7757 (+/-0.0037) |
| Brier Score | 0.156 |

**At Business Threshold (0.59):**
- Precision: 0.2272
- Recall: 0.5531
- Maximum Expected Profit: $36,819,000

**Confusion Matrix (threshold 0.5):**
- True Negatives: 41,995 | False Positives: 14,543
- False Negatives: 1,625 | True Positives: 3,340

### Top 10 Features (by importance)
1. EXT_SOURCE_MEAN (0.0771)
2. EXT_SOURCE_MAX (0.0380)
3. NAME_EDUCATION_TYPE (0.0174)
4. EXT_SOURCE_MIN (0.0161)
5. CODE_GENDER (0.0141)
6. FLAG_EMP_PHONE (0.0134)
7. CREDIT_TO_GOODS (0.0127)
8. PREV_REFUSAL_RATE (0.0118)
9. EXT_SOURCE_PRODUCT (0.0111)
10. NAME_INCOME_TYPE (0.0106)

### Visualizations (8 total, saved to `reports/Model Performance/`)
1. `confusion_matrix_baseline.png` - Confusion matrix (counts + percentages)
2. `roc_curve.png` - ROC curve with optimal threshold marker
3. `precision_recall_curve.png` - PR curve
4. `threshold_analysis.png` - Precision/Recall/F1 vs threshold
5. `profit_vs_threshold.png` - Expected profit vs threshold (U-shape)
6. `feature_importance.png` - Top 25 features by importance
7. `feature_importance_by_category.png` - Importance by category (pie chart)
8. `calibration_curve.png` - Probability calibration check

### Model Artifacts Saved
- `models/xgb_credit_model.pkl` - Trained XGBoost classifier (546 KB)
- `models/label_encoders.pkl` - 16 LabelEncoder objects
- `models/thresholds.pkl` - Statistical (0.5092) + business (0.59) thresholds
- `models/feature_names.pkl` - List of 211 feature names

### Database Output
- Full portfolio predictions stored in `portfolio_surveillance.db` (table: `loan_predictions`)

### Issues Encountered & Resolved
- [ISSUE-009] Spanish text translated to English

---

## 04_model_explainability.ipynb

**Date:** 2026-01-25
**Status:** Completed (44 cells)
**Location:** `notebooks/04_model_explainability.ipynb`
**Objective:** Explain XGBoost predictions using SHAP analysis for model interpretability, regulatory compliance, and business insights.

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Setup & Data Loading | Load model, encoders, thresholds from Notebook 03 |
| 2. Global Explainability | SHAP TreeExplainer on 10,000-sample subset |
| 2.5 Business Interpretation | Plain-English insights for top features |
| 3. Local Explainability | Waterfall plots for 3 case studies (FN, FP, TP) |
| 4. Dependence Plots | Top 5 features: value vs SHAP contribution |
| 5. Interaction Effects | SHAP interaction values for 1,000 samples |
| 6. Risk Segment Analysis | Feature importance by risk decile |
| 7. Regulatory Compliance | Model card, SR 11-7, fair lending, ECOA |
| 8. Executive Summary | Top 5 business insights + actionable recommendations |

### SHAP Global Importance (Top 10)
1. EXT_SOURCE_MEAN: 0.3905
2. ANNUITY_TO_CREDIT: 0.1216
3. CODE_GENDER: 0.1213
4. CREDIT_TO_GOODS: 0.1065
5. EXT_SOURCE_MAX: 0.0742
6. DAYS_BIRTH: 0.0623
7. FLAG_DOCUMENT_3: 0.0587
8. PREV_REFUSAL_RATE: 0.0533
9. EXT_SOURCE_3: 0.0506
10. EXT_SOURCE_MIN: 0.0481

### Case Studies (3 waterfall plots)
| Case | Type | Key Insight |
|------|------|-------------|
| False Negative | Missed default | All features appeared normal -- model limitation |
| False Positive | Over-rejection | Legitimate borrower flagged due to feature combination |
| True Positive | Correct rejection | Multiple strong risk indicators aligned |

### Visualizations (12 total, saved to `reports/SHAP Explainability/`)
1. `shap_summary_beeswarm.png` - Global SHAP beeswarm plot
2. `shap_global_importance.png` - Feature importance bar chart
3. `shap_waterfall_false_negative.png` - Local explanation (missed default)
4. `shap_waterfall_false_positive.png` - Local explanation (over-rejection)
5. `shap_waterfall_true_positive.png` - Local explanation (correct rejection)
6. `shap_dependence_ext_source_2.png` - External score vs SHAP
7. `shap_dependence_amt_credit.png` - Loan amount vs SHAP
8. `shap_dependence_age_years.png` - Age vs SHAP
9. `shap_dependence_amt_income.png` - Income vs SHAP
10. `shap_dependence_employment.png` - Employment duration vs SHAP
11. `shap_interaction_ext_employment.png` - Feature interaction plot
12. `shap_importance_by_segment.png` - Importance by risk decile

### Key Outputs
- `reports/executive_summary.txt` - Top 5 business insights + recommendations
- `reports/model_card.txt` - Full model documentation (SR 11-7 compliant)
- `reports/shap_feature_importance.csv` - SHAP importance values
- `models/shap_explainer_info.pkl` - SHAP explainer configuration

### Key Findings
- **External credit scores dominate** prediction explanations (SHAP importance 0.39)
- **Engineered ratios** (ANNUITY_TO_CREDIT, CREDIT_TO_GOODS) are more informative than raw amounts
- **Feature interactions** vary across risk segments (different drivers for high vs low risk)
- **Fairness concern**: CODE_GENDER and AGE_YEARS are protected attributes with significant SHAP impact
- **Model limitations**: False negatives show normal feature values across all dimensions

---

## 05_portfolio_surveillance.ipynb

**Date:** 2026-01-30 (updated 2026-02-01)
**Status:** Completed (37 cells)
**Location:** `notebooks/05_portfolio_surveillance.ipynb`
**Objective:** Build an autonomous AI agent (Claude Sonnet 4) that performs hierarchical portfolio surveillance, risk flagging, stress testing, and regulatory compliance reporting.

### AI Agent Configuration
- **Model:** Claude Sonnet 4 (`claude-sonnet-4-20250514`)
- **Persona:** Prudent Risk Officer
- **Max Turns:** 15
- **Tools:** 5 custom decorated functions
- **System Prompt:** 3,557 characters (hierarchical analysis protocol)

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Setup | Load model, encoders, data; initialize SQLite database |
| 2. Core Agent Architecture | Agent design philosophy, tool definitions, system prompt |
| 3. Hierarchical Analysis Pipeline | 4-phase analysis (Data Validation -> Risk Flagging -> Deep Dive -> Synthesis) |
| 3.1 Phase A | Data integrity validation, PSI distribution drift, freshness check |
| 3.4 Phase B | PD threshold breach detection, behavioral monitoring |
| 3.7 Phase C | Market intelligence (news), stress testing (3 scenarios), vintage analysis |
| 3.10 Phase D | Watch list generation, confidence ratings, recommendations |
| 4. Executive Reporting | VaR calculation, risk migration heatmap, top exposures |
| 5. Verification & Audit | Fair lending check, Basel IV validation, SR 11-7 documentation |
| 6. Demonstration | Example agent conversation, business impact summary |

### Tools Implemented
| Tool | Function | Purpose |
|------|----------|---------|
| `query_borrower_database` | `execute_sql_query()` | SQL access to loan_applications and engineered_features tables |
| `search_financial_news` | `FinancialNewsService` | Market intelligence from NewsAPI |
| `execute_risk_analysis` | `execute_risk_analysis()` | PD calculations, stress tests, threshold monitoring |
| `generate_report_section` | `generate_report_section()` | VaR, risk migration, top exposures, compliance reports |
| `log_audit_event` | `log_audit_event()` | Timestamped audit trail with JSON data |

### Key Metrics Produced
- **Portfolio Size:** 307,511 loans analyzed
- **High-Risk Flagged:** 59,874 borrowers (PD > 0.59)
- **Watch List:** 60,787 entries generated
- **Top 10 Expected Loss:** $11,786,710
- **Total Expected Loss:** $13.6 billion (portfolio-wide estimate)
- **Stress Test Impact:** +1.14% combined PD increase under adverse scenario

### Risk Tier Distribution
| Tier | Count | Description |
|------|-------|-------------|
| Green | 133,635 | Low risk |
| Yellow | 85,642 | Moderate risk |
| Orange | 56,746 | Elevated risk |
| Red | 31,488 | High risk |

### Compliance Validation Results
| Check | Status | Notes |
|-------|--------|-------|
| Data Integrity | PASS | All critical fields present |
| Basel IV Capital Adequacy | PASS | CET1/Tier1/Total capital requirements met |
| Fair Lending (ECOA) | FAIL | Disparate impact analysis requires review |
| SR 11-7 Documentation | PASS | Model card and validation complete |
| Verification (4 checks) | PASS | All 4 verification checks passed |

### Database Tables Created (SQLite: `portfolio_surveillance.db`)
1. `loan_applications` - Raw application data (307,511 x 122)
2. `engineered_features` - Engineered features (307,511 x 211)
3. `loan_predictions` - Model predictions with probabilities
4. `watch_list` - Flagged borrowers for monitoring

### Key Outputs
- `audit_trail.log` - Full audit trail with session IDs and timestamps
- `reports/ai_agent_overview.pptx` - PowerPoint presentation
- `portfolio_surveillance.db` - SQLite database (490 MB)

### Issues Encountered & Resolved
- [ISSUE-001] Tool decorator syntax (3 fix iterations)
- [ISSUE-002] JSON serialization with NumPy types
- [ISSUE-003] XGBoost dtype incompatibility in stress testing
- [ISSUE-008] Multiple agent session restarts

---

## Streamlit Dashboard (app.py)

**Date:** 2026-01-31
**Status:** Completed (296 lines)
**Location:** `app.py` (project root)
**Objective:** Interactive web dashboard for portfolio visualization, AI agent insights, and model performance metrics.

### Dashboard Structure

| Tab | Content |
|-----|---------|
| Portfolio Overview | 4 KPI cards (Total Loans, High Risk, Default Rate, Model Status); Risk distribution bar chart; Vintage analysis line chart |
| AI Agent Insights | Last analysis info; AI agent screenshot; 4-phase findings; Regulatory compliance cards (SR 11-7, Basel III/IV, IFRS 9) |
| Model Performance | 4 metric cards (AUC, Gini, KS, Brier); Model architecture details; Feature engineering summary; Business value; Tech stack |

### Sidebar Features
- Individual Loan Risk Calculator with 4 inputs
- Color-coded risk display (green/yellow/red)
- Expected Credit Loss calculation

### Key Metrics Displayed
- Total Loans: 307,511
- High-Risk Loans: 42,073 (13.68%)
- Default Probability: 8.07%
- ROC-AUC: 0.76 | Gini: 0.518 | KS: 0.421 | Brier: 0.156

---
