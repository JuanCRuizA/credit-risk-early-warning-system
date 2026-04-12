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

**Date:** 2026-01-25 (updated 2026-04-04)
**Status:** Completed (39 cells)
**Location:** `notebooks/02_FeatureEng.ipynb`
**Objective:** Transform raw data into 200+ engineered features integrating application, bureau, and previous application data.

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Setup and Data Loading | Load application_train.csv (307,511 x 122) |
| 2. Application Features | 23 engineered features (ratios, time-based, external scores, interactions, documents) |
| 3. Bureau Features | 41 features from credit bureau (loan counts, credit types, ratios) |
| Note: Unused Data Sources | Acknowledgment of installments_payments, POS_CASH_balance, credit_card_balance (excluded by scope) |
| 4. Previous Application Features | 30 features from previous applications (status, contracts, ratios) |
| 5. Merge All Features | Left joins: application + bureau + previous applications |
| 5.2b History Flags | HAS_BUREAU_HISTORY, HAS_PREV_APPLICATION binary flags |
| 6. Feature Selection | Final dataset preparation, missing value handling |
| 7. Summary | Feature count, save to CSV |

### Features Engineered (96 new features)

**From Application Train (23 features):**
- Ratio Features (6): DEBT_TO_INCOME, PAYMENT_BURDEN, CREDIT_TO_GOODS, ANNUITY_TO_CREDIT, INCOME_PER_PERSON, INCOME_TO_CREDIT
- Time-Based Features (6): AGE_YEARS, EMPLOYMENT_YEARS, FLAG_UNEMPLOYED, REGISTRATION_YEARS, ID_PUBLISH_YEARS, EMPLOYMENT_TO_AGE
- External Source Features (6): EXT_SOURCE_MEAN, EXT_SOURCE_WEIGHTED, EXT_SOURCE_PRODUCT, EXT_SOURCE_MIN, EXT_SOURCE_MAX, EXT_SOURCE_MISSING_COUNT
- Interaction Features (3): EXT_SCORE_x_PAYMENT_BURDEN, EXT_SCORE_x_AGE, EXT_SCORE_x_DEBT_RATIO
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

**History Flags (2 features):**
- HAS_BUREAU_HISTORY: Distinguishes "no bureau records" from "bureau records with zero values"
- HAS_PREV_APPLICATION: Distinguishes "no previous applications" from "previous applications with zero values"

### Key Outputs
- `data/processed/features_train.csv` - 307,511 rows x 218 columns
- `data/processed/feature_names.csv` - List of 216 feature names
- `reports/feature_correlations.png` - Top 20 features correlated with target

### Key Findings
- Final feature count: **218 columns** (122 original + 96 engineered)
- **99.4%** of applicants have bureau data (305,811 / 307,511)
- External source scores remain the most predictive features across all transformations
- Interaction features (EXT_SOURCE x financial ratios) add cross-domain signals
- Bureau features filled with 0 for applicants without credit history; HAS_BUREAU_HISTORY flag preserves the distinction
- EMPLOYMENT_YEARS set to NaN for unemployed/retired (consistent with NB01); signal captured by FLAG_UNEMPLOYED
- All 307,511 observations preserved through merges
- Unused data sources (installments_payments, POS_CASH_balance, credit_card_balance) documented as future iteration candidates

### Changes Log
- **2026-04-04**: Added temporal leakage assessment (DECISION-022):
  - Inserted new markdown cell before Section 3.1 bureau aggregations with full temporal analysis table
  - Documents why DAYS_CREDIT (always <= 0), CREDIT_ACTIVE, DAYS_CREDIT_ENDDATE, and BUREAU_CLOSED_LOAN_COUNT are temporally safe
  - Explains exclusion of bureau_balance.csv as the highest-risk leakage vector
  - Added inline temporal safety comments to the bureau aggregation code cell
  - Notebook now 39 cells (was 38)

---

## 03_model_training_evaluation.ipynb

**Date:** 2026-01-25 (updated 2026-04-04)
**Status:** Completed (48 cells)
**Location:** `notebooks/03_model_training_evaluation.ipynb`
**Objective:** Train an XGBoost credit risk model with business cost optimization, production-ready threshold selection, and agent-ready output artifacts.

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Setup and Data Loading | Load features_train.csv (307,511 x 213) |
| 2. Data Preparation | Label encoding for 16 categorical columns, stratified train/test split |
| 3. Train XGBoost Model | 200 estimators, scale_pos_weight=11.39 |
| 4. Cross-Validation | 5-Fold stratified CV: Mean AUC 0.7757 (+/-0.0037) |
| 5. Threshold Optimization | Youden's J (0.5092) and business optimal (0.79) |
| Currency disclaimer | Note on dataset currency units (not USD); ratios drive conclusions |
| 6. Business Cost Analysis | Data-driven cost parameters (median AMT_CREDIT), profit maximization |
| 7. Feature Importance | Top 25 features + category breakdown |
| 8.1 Model Calibration | Calibration curve for probability reliability |
| 8.1b Post-Hoc Calibration | Isotonic regression fix: preserves AUC, corrects inflated PDs |
| 8.2 Brier Score + Decile Table | Quantitative calibration: Brier Score, Brier Skill Score, per-decile predicted vs actual PD |
| 8.3 PSI: Train vs Test | Population Stability Index baseline for concept drift monitoring |
| 8.4 Risk Band Classification | 4-tier system (Low/Medium/High/Critical) aligned with NB05 |
| 8.5 Model Card JSON | Agent-ready JSON with all metadata, metrics, bands, PSI, calibration, limitations |
| 9. Save Final Model | Pickle files for model, encoders, thresholds, features |
| 10. Summary | Key metrics, portfolio predictions, database storage |

### Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.7793 |
| Gini Coefficient | 0.5585 |
| KS Statistic | 0.421 |
| Cross-Validation AUC | 0.7757 (+/-0.0037) |
| Brier Score | computed in Section 8.2 |
| PSI (train vs test) | computed in Section 8.3 |

**At Business Threshold (0.79):**
- Threshold interpretation: Only approve loans with P(default) < 0.21
- Conservative stance justified by 6:1 FN/FP cost asymmetry (LGD=60% / profit_margin=10%)

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

### Visualizations (11 total, saved to `reports/`)
1. `confusion_matrix_baseline.png` - Confusion matrix (counts + percentages)
2. `roc_curve.png` - ROC curve with optimal threshold marker
3. `precision_recall_curve.png` - PR curve
4. `threshold_analysis.png` - Precision/Recall/F1 vs threshold
5. `profit_vs_threshold.png` - Expected profit vs threshold (full range with green/red fill and peak marker)
6. `feature_importance.png` - Top 25 features by importance
7. `feature_importance_by_category.png` - Importance by category (pie chart)
8. `calibration_curve.png` - Probability calibration check (pre-calibration)
8b. `calibration_before_after.png` - Before/after isotonic calibration comparison
9. `psi_train_vs_test.png` - PSI dual-panel: overlay histogram + per-bin contribution
10. `risk_band_distribution.png` - Risk band bar chart with actual default rates

### Model Artifacts Saved
- `models/xgb_credit_model.pkl` - Trained XGBoost classifier (546 KB)
- `models/label_encoders.pkl` - 16 LabelEncoder objects
- `models/thresholds.pkl` - Statistical (0.5092) + business (calibrated) thresholds
- `models/calibrator.pkl` - Isotonic regression calibrator for PD correction
- `models/feature_names.pkl` - List of 211 feature names
- `models/model_card.json` - Agent-ready model card (metadata, metrics, risk bands, PSI, calibration, limitations)

### Database Output
- Full portfolio predictions stored in `portfolio_surveillance.db` (table: `loan_predictions`)

### Changes Log
- **2026-04-04**: Expanded regulatory coverage across four cells (DECISION-020):
  - Cell 0 (intro): Added EU AI Act Art. 10/13, FINMA 2017/1, FINMA 2023/1, Swiss nDSG to Regulatory Context; added Notebook Scope section (LIME, Kendall's τ, model card link); added Technical References section (Ribeiro et al. 2016; Lin & Wang 2025)
  - Cell 46 (Section 7 header): Added EU AI Act, FINMA 2017/1, FINMA 2023/1, Swiss nDSG to compliance bullet list
  - Cell 49 (Section 7.3 table): Added FINMA 2017/1 and FINMA 2023/1/Swiss nDSG rows to the regulation-SHAP mapping table
  - Cell 53 (executive summary): Added three new regulatory readiness checkmarks (FINMA 2017/1 & 2023/1, Swiss nDSG, EU AI Act Art. 10/13)
- **2026-04-04**: Created CLAUDE.md in project root with persistent context for Claude Code (project overview, tech stack, notebook status, working principles, regulatory references, language policy, session hygiene instructions)

### Original Changes Log
- **2026-04-04**: Expanded intro cell to match NB04 structure (DECISION-021):
  - Objectives updated to 7 items (added calibration, monitoring baseline, agent-ready artifacts)
  - Added Regulatory Context section: SR 11-7, Basel II/III CRR Art. 179, IFRS 9, GDPR Art. 22, ECOA, EU AI Act Art. 10/13, FINMA 2017/1, FINMA 2023/1, Swiss nDSG
  - Added Notebook Scope section mapping each analytical thread to its section number
  - Expanded Key Banking Context: added EL formula, 6:1 cost ratio, calibration as regulatory requirement
  - Added Technical References section: Fluss (2005), Verbraken (2014), Beque (2017), Yurdakul (2020)
- **2026-02-01**: [ISSUE-009] Spanish text translated to English
- **2026-04-02**: Added 9 new cells (8 new sections + 1 currency disclaimer):
  - Currency disclaimer before Section 6 (dataset units, not USD)
  - Section 6 updated: `LOAN_AMOUNT` now uses `df['AMT_CREDIT'].median()` instead of hardcoded $15,000 (DECISION-017)
  - Section 8.2: Brier Score + Decile Calibration Table (DECISION-016)
  - Section 8.3: PSI Train vs Test baseline (DECISION-015)
  - Section 8.4: Risk Band Classification System (DECISION-014)
  - Section 8.5: Agent-Ready Model Card JSON export (DECISION-014)
  - Summary cell updated with Brier Score, PSI, and model_card.json
- **2026-04-02**: Bibliographic references added as inline code comments (DECISION-018):
  - Cell 16: Fluss et al. (2005) — Youden's J statistic
  - Cell 23: Verbraken et al. (2014) — Profit-based classification
  - Cell 30: Bequé et al. (2017) — Calibration curve
  - Cell 32: Bequé et al. (2017) — Brier Score
  - Cell 34: Yurdakul & Naranjo (2020) — PSI
- **2026-04-02**: Fixed truncated threshold search range (ISSUE-014):
  - Extended `np.arange(0.05, 0.60, 0.02)` to `np.arange(0.05, 0.96, 0.02)`
  - Business-optimal threshold corrected from 0.59 to 0.79
  - Profit-vs-threshold plot improved: green/red fill, peak star marker, full x-axis range
- **2026-04-02**: Added post-hoc isotonic calibration (DECISION-019):
  - New Section 8.1b: Isotonic regression fixes inflated PDs from scale_pos_weight=11.39
  - Before/after calibration curve visualization
  - Calibrator saved as models/calibrator.pkl
  - PSI cell updated to use calibrated training predictions
  - Model card JSON updated with calibration method
  - Notebook now 48 cells (was 46)

---

## 04_model_explainability.ipynb

**Date:** 2026-01-25 (updated 2026-04-04)
**Status:** Completed (51 cells — calibrator now applied throughout)
**Location:** `notebooks/04_model_explainability.ipynb`
**Objective:** Explain XGBoost predictions using SHAP and LIME dual-method analysis for model interpretability, regulatory compliance, and business insights.

### Notebook Structure

| Section | Content |
|---------|---------|
| 1. Setup & Data Loading | Load model, encoders, thresholds from Notebook 03 |
| 2. Global Explainability | SHAP TreeExplainer on 10,000-sample subset |
| 2.5 Business Interpretation | Plain-English insights for top features |
| 3. Local Explainability | Waterfall plots for 3 case studies (FN, FP, TP) |
| 3.5 LIME Explainability | LIME local explanations for 3 case studies + SHAP vs LIME comparison |
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

### Visualizations (16 total, saved to `reports/`)
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
13. `lime_false_negative.png` - LIME local explanation (missed default)
14. `lime_false_positive.png` - LIME local explanation (over-rejection)
15. `lime_true_positive.png` - LIME local explanation (correct rejection)
16. `shap_vs_lime_comparison.png` - Side-by-side SHAP vs LIME feature attribution

### Key Outputs
- `reports/executive_summary.txt` - Top 5 business insights + recommendations
- `reports/model_card.txt` - Full model documentation (SR 11-7 compliant)
- `reports/shap_feature_importance.csv` - SHAP importance values
- `models/shap_explainer_info.pkl` - SHAP explainer configuration
- `reports/lime_false_negative.png` - LIME local explanation
- `reports/lime_false_positive.png` - LIME local explanation
- `reports/lime_true_positive.png` - LIME local explanation
- `reports/shap_vs_lime_comparison.png` - Method comparison chart

### Key Findings
- **External credit scores dominate** prediction explanations (SHAP importance 0.39)
- **Engineered ratios** (ANNUITY_TO_CREDIT, CREDIT_TO_GOODS) are more informative than raw amounts
- **Feature interactions** vary across risk segments (different drivers for high vs low risk)
- **Fairness concern**: CODE_GENDER and AGE_YEARS are protected attributes with significant SHAP impact
- **Model limitations**: False negatives show normal feature values across all dimensions
- **LIME validates SHAP**: Dual-method explainability confirms top feature attributions are robust across methodologies

### Changes Log
- **2026-04-04**: Applied isotonic calibrator throughout NB04 (DECISION-023):
  - Cell 3: Added `calibrator = joblib.load('calibrator.pkl')` alongside existing artifact loads
  - Cell 7: Applied `calibrator.predict()` to raw probabilities; defined `calibrated_predict_fn` wrapper for LIME
  - Cells 27, 28, 29: LIME `predict_fn` updated from `model.predict_proba` to `calibrated_predict_fn`
  - Cell 43: Risk decile probabilities now calibrated
  - Cell 47: Model card AUC and classification stats now use calibrated probabilities
  - All SHAP, LIME, and risk band outputs are now consistent with NB03 calibrated PD estimates

---

## 05_portfolio_surveillance.ipynb

**Date:** 2026-01-30 (updated 2026-04-08)
**Status:** Completed (39 cells)
**Location:** `notebooks/05_portfolio_surveillance.ipynb`
**Objective:** Build an autonomous AI agent (Claude Sonnet 4) that performs hierarchical portfolio surveillance, risk flagging, SHAP deep-dive, watch list generation, and regulatory compliance verification.

### AI Agent Configuration
- **Model:** Claude Sonnet 4 (`claude-sonnet-4-20250514`)
- **Persona:** Prudent Risk Officer
- **Max Turns:** 20 (increased from 15 after Run 1 consumed turns on failed SHAP attempts)
- **Tools:** 5 custom JSON-schema tool definitions (not `@tool` decorators — see DECISION-024 reference)
- **Framework:** ReAct (Yao et al. 2023, arXiv:2210.03629) — Thought → Action → Observation cycles
- **System Prompt:** hierarchical 5-phase analysis protocol

### Notebook Structure (39 cells)

| Cell | Content |
|------|---------|
| 0 | Full intro: series position, 5-tool table, 5-phase pipeline, run metrics, regulatory table |
| 1–6 | Setup, SQLite load, tool definitions, system prompt, agent loop |
| 7 | Architecture Decision markdown (TOOL_SCHEMAS vs @tool) |
| 8–35 | Phase A → D execution, SHAP branch, stress tests, watch list |
| 36 | Phase E VerificationSubagent trigger (post-agent Python code) |
| 37 | Clean report display cell (post-processes agent_result for Cell 6.3) |
| 38 | Business Impact Summary markdown (closing, 7-framework regulatory table) |

### 5-Phase Analysis Protocol
| Phase | Focus | Key Output |
|-------|-------|------------|
| A | Data Integrity | 307,511 records validated, missing rates, freshness |
| B | Risk Flagging | 7,370 borrowers at PD > 0.79 business threshold |
| C | SHAP Deep-Dive | Top 5 highest-PD borrowers, stress scenarios |
| D | Watch List & Recommendations | 16,045 entries, 4 strategic actions |
| E | Verification & Compliance | 4/4 checks PASS (Basel IV, EL, VaR, rankings) |

### Tools Implemented
| Tool | Function | Purpose |
|------|----------|---------|
| `query_borrower_database` | `execute_sql_query()` | SQL access to loan_applications and engineered_features tables |
| `search_financial_news` | `FinancialNewsService` | Market intelligence (mocked stub; production = NewsAPI/Finnhub) |
| `execute_risk_analysis` | `execute_risk_analysis()` | PD calculations, SHAP branch, stress tests, threshold monitoring |
| `generate_report_section` | `generate_report_section()` | VaR, risk migration, top exposures, compliance reports |
| `log_audit_event` | `log_audit_event()` | Timestamped audit trail with JSON data |

### Run 2 Metrics (Live execution, 2026-04-08)
- **Turns:** 20 | **Tool calls:** ~16 | **Cost:** $0.6696 est.
- **Tokens:** 167,128 input / 11,217 output
- **High-Risk Flagged:** 7,370 borrowers (PD > 0.79 business threshold)
- **Watch List:** 16,045 entries | **Combined EL:** $3.88B
- **Phase E:** 4/4 PASS
- **PSI (EXT_SOURCE_3):** 0.28 — Significant Shift, model recalibration recommended

### Risk Tier Distribution (Run 2)
| Tier | Count | Description |
|------|-------|-------------|
| Green (PD < 0.30) | 136,626 | Low risk |
| Yellow (0.30–0.52) | 83,308 | Moderate risk |
| Orange (0.52–0.79) | 56,074 | Elevated risk |
| Red (PD > 0.79) | 31,503 | High risk |

### Compliance Validation Results
| Check | Status | Notes |
|-------|--------|-------|
| Data Integrity | PASS | All critical fields present |
| Basel IV Capital Adequacy | PASS | Required capital $652.8M, RWA $8.16B |
| Fair Lending (ECOA) | Requires Review | Term "age" in risk narrative; confirm credit-neutral use |
| SR 11-7 Documentation | COMPLIANT | Model card and validation complete |
| Phase E Verification (4 checks) | PASS | EL, VaR, percentages, ranking all verified |

### What Is Real vs Mocked
- **SQL queries:** Real (SQLite, 307,511 rows)
- **SHAP:** Real (TreeExplainer on in-memory XGBoost; numeric-only filter applied)
- **Financial news:** Mocked (structured stub; production would use NewsAPI/Finnhub)
- **Stress tests:** Real (executed via `exec()` with df_features)

### Database Tables Created (SQLite: `portfolio_surveillance.db`)
1. `loan_applications` - Raw application data (307,511 x 122)
2. `engineered_features` - Engineered features (307,511 x 216)
3. `loan_predictions` - Model predictions with probabilities
4. `watch_list` - Flagged borrowers for monitoring

### Key Outputs
- `audit_trail.log` - Full audit trail with session IDs and timestamps
- `reports/agent_output_latest.json` - Structured April 8 run data (loaded by dashboard)
- `portfolio_surveillance.db` - SQLite database (490 MB)

### Issues Encountered & Resolved
- [ISSUE-001] Tool decorator syntax (3 fix iterations)
- [ISSUE-002] JSON serialization with NumPy types
- [ISSUE-003] XGBoost dtype incompatibility in stress testing
- [ISSUE-008] Multiple agent session restarts

### Changes Log
- **2026-04-08**: Run 2 live execution completed; SHAP alignment fix applied (numeric-only filter on X_calc)
- **2026-04-08**: Phase E VerificationSubagent added (Cell 36) — judge-agent pattern (Okpala et al. 2025, arXiv:2502.05439)
- **2026-04-08**: Cell 37 added — clean report display post-processor for Cell 6.3 Tee capture fallback
- **2026-04-08**: Cell 0 updated to 5-phase protocol; max_turns increased to 20; Cell 38 Business Impact Summary added
- **2026-04-08**: `reports/agent_output_latest.json` created as structured export of Run 2 metrics for dashboard

---

## Streamlit Dashboard (app.py)

**Date:** 2026-01-31 (updated 2026-04-11)
**Status:** Completed — Sprint A/B/C applied (~1,400+ lines)
**Location:** `app.py` (project root)
**Objective:** Interactive web dashboard for portfolio visualization, model performance, dual explainability, AI agent insights, and regulatory compliance.

### Dashboard Structure (5 tabs)

| Tab | Content |
|-----|---------|
| 1 — Portfolio Overview | Hero paragraph; Executive Brief expander (loads `executive_summary.txt`); 4 KPI cards (Total Loans, High Risk, Default Rate, Model Status); Risk Band table from `model_card.json`; Risk distribution bar chart; Vintage analysis line chart |
| 2 — Model Performance | 4-metric row (AUC/Gini/KS/Brier from `model_card.json`); Decile/lift table; `calibration_before_after.png`; ROC curve with live vs. calibrated AUC legend note |
| 3 — SHAP Explainability | Global SHAP beeswarm + importance bar chart; 3 waterfall case studies (FN, FP, TP); LIME Validation section (3 LIME plots + `shap_vs_lime_comparison.png`) |
| 4 — AI Agent Insights | Fully dynamic from `reports/agent_output_latest.json`; Phase A–D KPIs and findings; PSI table with color coding; Agent reasoning trace expander; Basel IV compliance status badge; graceful fallback if JSON absent |
| 5 — Regulatory Compliance | 8-framework compliance table; audit log download button; model card JSON download button; adverse action notice sample in expander |

### Sidebar Features
- Individual Loan Risk Calculator (4 inputs: credit score, age, loan amount, monthly payment)
- Color-coded risk display (green/yellow/red) at business threshold 0.79
- Expected Credit Loss (EL = PD × LGD × EAD) calculation
- Live SHAP top-3 risk drivers + top-2 protective factors (Sprint B7 — DECISION-026)
- "Technical Architecture" expander (full 4-layer pipeline)

### Key Metrics Displayed (sourced from `model_card.json` and `agent_output_latest.json`)
- Total Loans: 307,511 | High-Risk (PD > 0.79): 7,370 (11.98%) | Avg PD: 8.07%
- ROC-AUC: 0.7778 | Gini: 0.5556 | KS: computed live | Brier: 0.0668
- 5-Fold CV AUC: 0.7755 ± 0.0035

### Changes Log
- **2026-04-11 Sprint A**: Hero paragraph, Executive Brief expander, Risk Band table, 4-metric KPI row, decile/lift table from model_card.json, calibration_before_after.png, ROC legend with live vs. calibrated AUC note
- **2026-04-11 Sprint B**: Dynamic Tab 4 (AI Agent) from agent_output_latest.json; LIME Validation section in Tab 3 (SHAP); live SHAP attribution in sidebar (DECISION-026); audit log + model card download buttons in Tab 5; adverse action notice expander; Tab order swapped (SHAP → Tab 3, AI Agent → Tab 4 — DECISION-025)
- **2026-04-11 Sprint C**: "About & Methods" renamed to "Technical Architecture" with full pipeline description
- **2026-04-11 Bug fixes**: pandas `applymap` → `.map()` (ISSUE-015); `encoding='utf-8'` on all text file reads (ISSUE-016); interaction features computed in risk calculator (ISSUE-017)

---
