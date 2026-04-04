# Credit Risk Early Warning System (CREWS) - Technical Decisions Log

## Purpose
Document key technical decisions, rationale, and alternatives considered during development.

---

## Quick Reference

### Active Decisions
- [DECISION-001] Home Credit Dataset Selection
- [DECISION-002] Comprehensive EDA Before Feature Engineering
- [DECISION-003] Multi-Source Feature Engineering (Application + Bureau + Previous Apps)
- [DECISION-004] XGBoost Over Other Algorithms
- [DECISION-005] Label Encoding Over One-Hot Encoding for Categoricals
- [DECISION-006] Stratified Train/Test Split Over Temporal Split
- [DECISION-007] scale_pos_weight for Class Imbalance Handling
- [DECISION-008] Dual Threshold Strategy (Statistical + Business)
- [DECISION-009] SHAP TreeExplainer + LIME for Dual Model Explainability
- [DECISION-010] Autonomous AI Agent with Claude Sonnet 4
- [DECISION-011] SQLite for Portfolio Surveillance Database
- [DECISION-012] Streamlit with Plotly for Dashboard Deployment
- [DECISION-013] XGBoost Native Missing Value Handling Over Imputation
- [DECISION-014] Agent-Ready Model Card JSON + Risk Band Classification
- [DECISION-015] PSI Baseline for Concept Drift Monitoring
- [DECISION-016] Calibration-Enhanced Evaluation (Brier Score + Decile Table)
- [DECISION-017] Data-Driven Business Cost Parameters (Real AMT_CREDIT Median)
- [DECISION-018] Bibliographic References as Inline Code Comments in NB03
- [DECISION-019] Post-Hoc Isotonic Calibration for Regulatory-Compliant PD Estimates
- [DECISION-020] Expanded Regulatory Coverage in NB04 (FINMA, Swiss nDSG, EU AI Act)

### Pending Review
- None

### Rejected
- None

---

## Decisions

### [DECISION-001] Home Credit Dataset Selection
**Date:** 2026-01-25
**Status:** Implemented
**Context:** Need a credit risk dataset that demonstrates real-world banking scenarios for a portfolio project targeting Banking Data Scientist roles.
**Decision:** Use the Home Credit Default Risk dataset from Kaggle.
**Rationale:**
- 307,511 loan applications with 122 features -- realistic scale
- Multiple supplementary tables (bureau, previous applications, installments) enable rich feature engineering
- 8.07% default rate mirrors real retail banking portfolios (5-15% typical range)
- Includes external credit bureau scores (EXT_SOURCE_1, 2, 3) -- key predictors in real banking
- Well-documented with column descriptions
- Industry-relevant: Home Credit is an actual financial services provider
**Alternatives Considered:**
- IEEE-CIS Fraud Detection: Fraud detection focus, not credit risk; previously used in BAFS project
- German Credit Dataset: Too small (1,000 records), outdated
- Lending Club: US-only, peer-to-peer lending (not traditional banking)
- Synthetic data: Less credible for portfolio demonstration
**Consequences:**
- Large dataset enables robust model training with 200K+ training samples
- Multiple supplementary tables require complex merge operations
- Class imbalance (8% default) needs handling via scale_pos_weight
- Missing data in 67 columns requires documented handling strategy
**Related:** `notebooks/01_eda.ipynb`

---

### [DECISION-002] Comprehensive EDA Before Feature Engineering
**Date:** 2026-01-25
**Status:** Implemented
**Context:** Need to understand data quality, distributions, and relationships before building features and models.
**Decision:** Perform thorough EDA covering target analysis, missing values (patterns + strategy), correlations, outliers, and categorical analysis across all 10 CSV files.
**Rationale:**
- Identifies data quality issues early (DAYS_EMPLOYED anomaly, 67 columns with missing data)
- Missing value pattern analysis reveals shared data sources (housing features missing together)
- Correlation heatmap identifies multicollinearity (AMT_CREDIT vs AMT_GOODS_PRICE: r=0.99)
- Outlier analysis prevents data quality issues from propagating to models
- EXT_SOURCE features identified as top predictors, guiding feature engineering priority
- Documented null handling strategy provides clear guidance for Notebook 02
**Alternatives Considered:**
- Minimal EDA (just target + basic stats): Would miss critical data quality issues
- Jump directly to modeling: Would produce suboptimal features and miss anomalies
- Automated EDA tools (pandas-profiling, sweetviz): Less customizable, generic output
**Consequences:**
- 61-cell notebook with 10 visualizations saved to `reports/`
- Formal null handling strategy: Very high (>70%) DROP, High (40-70%) KEEP for XGBoost, Moderate/Low impute or KEEP
- Key insight: XGBoost handles missing values natively -- reduces imputation burden
**Related:** `notebooks/01_eda.ipynb`, `docs/issues-solutions.md` [ISSUE-010]

---

### [DECISION-003] Multi-Source Feature Engineering (Application + Bureau + Previous Applications)
**Date:** 2026-01-25
**Status:** Implemented
**Context:** The raw application_train.csv has 122 columns, but credit risk prediction benefits significantly from enrichment with external credit history and previous application patterns.
**Decision:** Engineer 96 new features from three data sources:
- 23 features from application data (ratios, time-based, external scores, interactions, documents)
- 41 features from bureau data (loan counts, credit types, overdue ratios)
- 30 features from previous applications (approval rates, contract types, credit ratios)
- 2 history flags (HAS_BUREAU_HISTORY, HAS_PREV_APPLICATION)
**Rationale:**
- **Ratio features** (DEBT_TO_INCOME, PAYMENT_BURDEN) are more informative than raw amounts -- they normalize for income and loan size differences
- **EXT_SOURCE combinations** (mean, weighted, product, min, max) capture multi-bureau signal better than individual scores
- **Interaction features** (EXT_SOURCE x PAYMENT_BURDEN, x AGE, x DEBT_RATIO) combine cross-domain signals that neither source captures alone
- **Bureau history** provides credit behavior context (active loans, overdue ratios, debt-to-credit)
- **Previous application patterns** reveal relationship history with the lender (approval rate, refusal rate)
- **FLAG_UNEMPLOYED** captures the DAYS_EMPLOYED anomaly as a binary feature
- **History flags** distinguish "no bureau/previous history" from "history with zero values" -- critical for AI agent narratives
- **EXT_SOURCE_WEIGHTED** weights (0.2/0.4/0.4) derived from EDA correlations: r = -0.155, -0.161, -0.179
- Final feature count (216) provides rich signal for XGBoost without excessive dimensionality
**Alternatives Considered:**
- Application features only (122 columns): Misses bureau and historical signal
- All 434 raw columns without engineering: High dimensionality, redundancy, no domain insight
- Automated feature engineering (Featuretools, tsfresh): Less domain control, harder to explain to regulators
- Include installments_payments and credit_card_balance: Added complexity for marginal gains (deferred and documented)
**Consequences:**
- Feature count increases from 122 to 218 (including SK_ID_CURR and TARGET)
- `features_train.csv` output (large but manageable)
- 99.4% of applicants have bureau data (good coverage)
- Bureau/previous features filled with 0 for applicants without history; HAS_BUREAU_HISTORY/HAS_PREV_APPLICATION flags preserve the distinction
- EMPLOYMENT_YEARS set to NaN for unemployed/retired (consistent with NB01 EDA; signal captured by FLAG_UNEMPLOYED)
- Division-by-zero handled: ratio features produce NaN when denominator is zero (XGBoost handles natively)
- Unused data sources (installments_payments, POS_CASH_balance, credit_card_balance) documented as future iteration candidates
**Related:** `notebooks/02_FeatureEng.ipynb`

---

### [DECISION-004] XGBoost Over Other Algorithms
**Date:** 2026-01-25
**Status:** Implemented
**Context:** Need a model that handles class imbalance, missing values, non-linear relationships, and provides feature importance -- while maintaining acceptable performance on 307K records.
**Decision:** Deploy XGBoost (Gradient Boosted Decision Trees) as the production model.
**Rationale:**
- **AUC-ROC: 0.7793** with **Gini: 0.5585** -- strong performance for retail credit risk
- **Native missing value handling**: XGBoost learns optimal split direction for NaN values, critical with 67 columns having missing data
- **Class imbalance**: Built-in `scale_pos_weight` parameter (set to 11.39)
- **Feature importance**: Tree-based importance + SHAP compatibility for regulatory explainability
- **Cross-validation stability**: Mean AUC 0.7757 (+/-0.0037) -- very low variance across 5 folds
- **Industry standard**: XGBoost is the dominant algorithm in credit risk modeling
- **Inference speed**: <1ms per prediction -- suitable for real-time scoring
- **Robust to outliers**: Tree splits are rank-based, not affected by extreme values
**Alternatives Considered:**
- Logistic Regression: More interpretable but limited to linear relationships; lower performance expected
- LightGBM: Comparable performance, could be explored as challenger model
- Random Forest: Typically slower, less performant than boosted trees
- Neural Networks: Overkill for tabular data, harder to explain for SR 11-7 compliance
- CatBoost: Native categorical handling useful but XGBoost with label encoding performs well
**Consequences:**
- Model file: 546 KB (`xgb_credit_model.pkl`) -- lightweight for deployment
- Training time: ~30 seconds on full dataset
- Requires label encoding for 16 categorical columns (see DECISION-005)
- Compatible with SHAP TreeExplainer for exact Shapley values (see DECISION-009)
**Related:** `notebooks/03_model_training_evaluation.ipynb`

---

### [DECISION-005] Label Encoding Over One-Hot Encoding for Categoricals
**Date:** 2026-01-25
**Status:** Implemented
**Context:** 16 categorical columns need numeric encoding for XGBoost. The choice between label encoding and one-hot encoding affects dimensionality, memory, and model interpretation.
**Decision:** Use LabelEncoder for all 16 categorical columns. Store encoders in `label_encoders.pkl` for production inference.
**Rationale:**
- **XGBoost handles ordinal encoding well**: Tree-based models can split on any encoded value, so arbitrary integer encoding doesn't create false ordinal relationships
- **Dimensionality control**: One-hot encoding ORGANIZATION_TYPE (58 categories) alone would add 58 columns; label encoding keeps 1 column
- **Memory efficiency**: 16 columns vs potentially 120+ columns with one-hot encoding
- **Consistency**: All categoricals treated uniformly, simple to maintain in production
**Alternatives Considered:**
- One-hot encoding: Explodes dimensionality (especially ORGANIZATION_TYPE with 58 categories)
- Target encoding: Risk of data leakage if not implemented with proper cross-validation folds
- CatBoost with native categorical handling: Requires switching model framework
- Ordinal encoding with meaningful order: Not all features have natural order (e.g., ORGANIZATION_TYPE)
**Consequences:**
- 16 LabelEncoder objects stored in `models/label_encoders.pkl`
- Must apply same encoding at inference time (encoders must handle unseen categories)
- SHAP values reference encoded integers, not original category names (requires mapping for interpretation)
**Related:** `notebooks/03_model_training_evaluation.ipynb` (Section 2)

---

### [DECISION-006] Stratified Train/Test Split Over Temporal Split
**Date:** 2026-01-25
**Status:** Implemented
**Context:** Need to split data for model training and evaluation. The Home Credit dataset has no explicit temporal ordering (unlike the BAFS project which used TransactionDT).
**Decision:** Use stratified 80/20 train/test split preserving the 8.07% default rate in both splits.
**Rationale:**
- **No clear temporal column**: The Home Credit dataset does not have a transaction timestamp suitable for temporal splitting (unlike fraud detection data)
- **Class preservation**: Stratification ensures both train and test sets maintain the 8.07% default rate, preventing biased evaluation
- **Standard practice**: For credit scoring models without temporal ordering, stratified random splits are industry standard
- **Cross-validation supplements**: 5-Fold CV provides additional robustness assessment (Mean AUC: 0.7757 +/- 0.0037)
**Alternatives Considered:**
- Temporal split on DAYS_BIRTH or DAYS_EMPLOYED: These are applicant attributes, not application dates
- Simple random split (no stratification): Could produce imbalanced splits with only 8% positive class
- K-Fold only (no holdout): No final independent test set for reporting
**Consequences:**
- X_train: 246,008 rows; X_test: 61,503 rows
- Default rates preserved: 8.07% in both sets
- No temporal concept drift testing (acceptable given dataset structure)
- More optimistic than temporal split (feature benefit, not a concern without time ordering)
**Related:** `notebooks/03_model_training_evaluation.ipynb` (Section 2)

---

### [DECISION-007] scale_pos_weight for Class Imbalance Handling
**Date:** 2026-01-25
**Status:** Implemented
**Context:** With 8.07% default rate (class imbalance ratio 11.4:1), the model needs to be sensitized to the minority class (defaults) to achieve adequate recall.
**Decision:** Set `scale_pos_weight = 11.39` (ratio of non-defaults to defaults: 282,686 / 24,825).
**Rationale:**
- **Built-in XGBoost parameter**: No additional preprocessing or sampling required
- **Equivalent to class weighting**: Makes each default observation count 11.39x during gradient computation
- **Simple and effective**: Directly addresses the 1:11.4 imbalance ratio
- **No data modification**: Unlike SMOTE or undersampling, the training data remains unchanged
- **Interpretable**: The weight directly reflects the class distribution
**Alternatives Considered:**
- SMOTE (oversampling minority class): Creates synthetic samples that may not represent real defaults; increases training time
- Random undersampling (majority class): Discards valuable data (would lose ~258K legitimate samples)
- No handling (default parameters): Model would predict "no default" for almost everything, achieving 92% accuracy but useless recall
- Custom sample weights: More flexible but scale_pos_weight is equivalent for binary classification
**Consequences:**
- Recall improves significantly vs no imbalance handling
- May slightly increase false positive rate (trade-off for catching more defaults)
- Business threshold optimization (DECISION-008) further refines the precision-recall trade-off
**Related:** `notebooks/03_model_training_evaluation.ipynb` (Section 3)

---

### [DECISION-008] Dual Threshold Strategy (Statistical + Business)
**Date:** 2026-01-25 (updated 2026-04-02)
**Status:** Implemented
**Context:** The default threshold (0.5) is arbitrary. Credit risk models need thresholds optimized for both statistical performance and business value.
**Decision:** Calculate two optimal thresholds:
- **Statistical optimal (Youden's J):** 0.5092 -- maximizes sensitivity + specificity
- **Business optimal:** 0.79 -- maximizes expected profit using cost parameters
**Rationale:**
- **Two separate concerns**: Statistical performance and business value often conflict
- **Youden's J** provides the mathematically optimal balance of sensitivity/specificity
- **Profit optimization** incorporates data-driven business costs (updated 2026-04-02):
  - Average loan: $513,531 (median AMT_CREDIT from dataset; see DECISION-017)
  - Loss Given Default (LGD): 60% (Basel IRB Foundation, unsecured consumer credit)
  - Cost of default (FN): $308,119
  - Profit per good loan (FP cost): $51,353
  - FN/FP cost ratio: 6:1 (LGD/profit_margin = 0.60/0.10)
- **Business threshold** maximizes expected profit across the full [0.05, 0.95] range
- Both thresholds stored in `models/thresholds.pkl` for deployment flexibility
- **2026-04-02 correction:** Previous threshold of 0.59 was a truncation artifact -- the search range `np.arange(0.05, 0.60, 0.02)` stopped before the true peak. Extended to 0.96 revealed the real optimum at 0.79. See ISSUE-014.
**Alternatives Considered:**
- Fixed threshold (0.5): Arbitrary, not optimized for either statistical or business criteria
- Single threshold (statistical only): Ignores business economics
- Multi-threshold strategy (like BAFS auto-block/manual review/auto-approve): Considered but simpler binary decision is more appropriate for credit approval (approve/reject)
- Cost-sensitive learning (custom loss function): More complex, harder to validate
**Consequences:**
- At business threshold (0.79): Only loans with P(default) < 0.21 are approved -- conservative stance justified by 6:1 cost asymmetry
- The 6:1 FN/FP cost ratio (LGD=60% / profit_margin=10%) means one bad loan wipes out the profit from six good ones, pushing the optimal threshold high
- Decision shifts from "catch all defaults" to "maximize portfolio profitability"
**Related:** `notebooks/03_model_training_evaluation.ipynb` (Sections 5-6), ISSUE-014

---

### [DECISION-009] SHAP TreeExplainer + LIME for Dual Model Explainability
**Date:** 2026-01-25 (updated 2026-03-28)
**Status:** Implemented
**Context:** Banking regulators (SR 11-7), customers, and analysts need to understand why the model approves or rejects each loan. A "black box" model is not deployable in regulated financial services.
**Decision:** Use SHAP TreeExplainer as the primary explainability method (exact Shapley values) complemented by LIME (model-agnostic local linear approximations) for independent validation of feature attributions.
**Rationale:**
- **Exact values**: TreeExplainer computes exact (not approximate) Shapley values for tree-based models
- **Speed**: Polynomial-time algorithm handles 10,000 samples in seconds with 211 features
- **Theory-grounded**: Game-theoretic foundation provides mathematically rigorous feature attribution
- **Regulatory acceptance**: SHAP is the industry standard for model explainability in banking
- **Dual-purpose**: Global plots (beeswarm, bar) for stakeholders; local plots (waterfall) for analysts and auditors
- **Dependence plots**: Reveal non-linear relationships and feature interactions
- **Segment analysis**: Feature importance varies by risk decile -- actionable for portfolio management
- **LIME complements SHAP**: Provides independent, model-agnostic validation using perturbation-based local linear models
- **Agreement strengthens confidence**: When SHAP and LIME agree on top features, explanations are robust to methodology choice
- **Disagreement is informative**: Highlights cases where linear approximation breaks down, flagging complex feature interactions
- **Dual-method differentiator**: Exceeds single-method requirements (EU AI Act Art. 13, FINMA Guidance 03/2024)
**Alternatives Considered:**
- LIME alone (without SHAP): Approximation-based, less precise for trees; now used as complementary method rather than replacement
- XGBoost built-in `feature_importances_`: Only global importance (gain/weight), no local explanations
- Partial Dependence Plots alone: Show marginal effects but don't explain individual decisions
- Permutation importance: Computationally expensive, no local explanations
**Consequences:**
- 12 publication-ready visualizations saved to `reports/SHAP Explainability/`
- Executive summary with top 5 business insights generated
- Model card with SR 11-7 compliance documentation produced
- Fair lending concern identified: CODE_GENDER has significant SHAP contribution
- SHAP analysis reveals EXT_SOURCE_MEAN dominates predictions (SHAP importance 0.39)
- 4 additional LIME visualizations saved to `reports/` (3 case studies + comparison chart)
- LIME explainer requires `lime>=0.2.0` package (added to requirements.txt)
- Dual-method approach adds ~30 seconds to notebook runtime for 3 LIME explanations
- CREWS now uses: XGBoost + SHAP + LIME (vs SAFE project: LightGBM + SHAP only)
**Related:** `notebooks/04_model_explainability.ipynb`, `reports/model_card.txt`, `reports/executive_summary.txt`

---

### [DECISION-010] Autonomous AI Agent with Claude Sonnet 4
**Date:** 2026-01-30
**Status:** Implemented
**Context:** Portfolio surveillance in banking requires continuous monitoring, risk flagging, stress testing, and compliance reporting. Automating this with an AI agent demonstrates production-ready autonomous analysis capability.
**Decision:** Build an autonomous AI agent using Anthropic's Claude Sonnet 4 API with the Claude Agent SDK, implementing a 4-phase hierarchical analysis protocol with 5 custom tools.
**Rationale:**
- **Autonomous analysis**: Agent performs multi-step analysis without manual intervention (15-turn conversation)
- **Hierarchical protocol**: 4 phases mirror real banking surveillance workflows (Validation -> Flagging -> Deep Dive -> Synthesis)
- **Tool use**: 5 tools enable database queries, risk calculations, news search, report generation, and audit logging
- **Regulatory compliance**: Built-in fair lending checks, Basel IV validation, SR 11-7 documentation
- **Audit trail**: Every action logged with timestamps, session IDs, and data summaries
- **Claude Sonnet 4**: Reliable reasoning for financial analysis, supports tool use natively
**Alternatives Considered:**
- LangChain agents with OpenAI: More established ecosystem but less control over agent behavior
- Simple script (no AI agent): Would demonstrate analysis but not autonomous decision-making
- Multi-agent system (CrewAI, AutoGen): Added complexity without proportional benefit for single-portfolio analysis
- GPT-4 based agent: Comparable capability but Anthropic chosen for consistency with Claude Code workflow
**Consequences:**
- Requires `ANTHROPIC_API_KEY` in `.env` for agent execution
- Agent identified 59,874 high-risk borrowers and generated 60,787 watch list entries
- Fair lending check returned FAIL (disparate impact requires review) -- genuine finding, not a bug
- Stress testing showed +1.14% combined PD increase under adverse scenario
- Multiple agent sessions needed during development (ISSUE-001 through ISSUE-003 resolved)
- `audit_trail.log` provides full compliance record of agent actions
**Related:** `notebooks/05_portfolio_surveillance.ipynb`, `audit_trail.log`

---

### [DECISION-011] SQLite for Portfolio Surveillance Database
**Date:** 2026-01-30
**Status:** Implemented
**Context:** The AI agent needs structured access to loan data, predictions, and watch lists. A database enables SQL-based queries that are more flexible than DataFrame operations.
**Decision:** Use SQLite as the local database for portfolio surveillance, storing application data, engineered features, predictions, and watch lists.
**Rationale:**
- **Zero configuration**: No database server required (file-based)
- **SQL access**: Agent can write flexible queries via the `query_borrower_database` tool
- **Portable**: Database file (`portfolio_surveillance.db`, 490 MB) is self-contained
- **Python integration**: Native `sqlite3` module, no additional dependencies
- **Suitable scale**: 307,511 rows x 200+ columns is well within SQLite's capabilities
- **Audit-friendly**: Structured storage with clear table separation
**Alternatives Considered:**
- PostgreSQL: Overkill for single-user portfolio analysis; requires server setup
- pandas DataFrames only: No SQL flexibility; memory constraints with multiple large DataFrames
- DuckDB: Faster for analytics but less common; SQLite is more universally understood
- CSV files: No query capability; slow for large-scale filtering
**Consequences:**
- Database file is 490 MB (not committed to git; created by running notebooks)
- 4 main tables: loan_applications, engineered_features, loan_predictions, watch_list
- Agent can run arbitrary SQL queries with `LIMIT 1000` safety guard
- Data persists across notebook sessions (no need to re-process)
**Related:** `notebooks/05_portfolio_surveillance.ipynb`, `data/portfolio_surveillance.db`

---

### [DECISION-012] Streamlit with Plotly for Dashboard Deployment
**Date:** 2026-01-31
**Status:** Implemented
**Context:** Need an interactive dashboard to present portfolio analytics, AI agent insights, and model performance to stakeholders and recruiters.
**Decision:** Build a Streamlit app (`app.py`) with Plotly charts, organized into 3 tabs with a sidebar risk calculator.
**Rationale:**
- **Rapid development**: Streamlit enables dashboard creation in ~300 lines of Python
- **Plotly interactivity**: Hover tooltips, zooming, and dynamic charts enhance user experience
- **Free cloud deployment**: Streamlit Cloud provides free hosting with GitHub integration
- **Tab organization**: Portfolio Overview, AI Agent Insights, and Model Performance cover all key audiences
- **Sidebar calculator**: Demonstrates real-time risk scoring capability
- **Professional aesthetic**: Clean layout with metric cards, charts, and compliance sections
**Alternatives Considered:**
- Dash by Plotly: More flexible but more complex; requires Flask backend knowledge
- Jupyter Voila: Limited interactivity compared to Streamlit
- PowerBI/Tableau: Requires separate licenses; not Python-native
- Flask + HTML: Maximum flexibility but significantly more development time
**Consequences:**
- Single-file app (`app.py`, 296 lines) -- easy to maintain and deploy
- Risk calculator uses simplified model (mock prediction) -- not production inference
- Dashboard displays pre-computed metrics from notebooks (not live model scoring)
- Deployable to Streamlit Cloud with `requirements.txt`
**Related:** `app.py`, README.md (Live Demo section)

---

### [DECISION-013] XGBoost Native Missing Value Handling Over Imputation
**Date:** 2026-01-25
**Status:** Implemented
**Context:** 67 of 122 columns have missing data (up to 70% missing in some columns). Need a strategy that preserves information while maintaining model quality.
**Decision:** Rely primarily on XGBoost's native missing value handling. Only fill bureau and previous application aggregation features with 0 (no history = zero count). Do not impute other missing values.
**Rationale:**
- **XGBoost learns optimal splits for NaN**: At each tree node, missing values are routed to the child that minimizes loss -- this is learned during training, not assumed
- **Imputation destroys information**: Filling NaN with median/mean removes the signal that "this value was not available" which itself may be predictive
- **Documented strategy** from EDA (Section 4.2):
  - Very high missing (>70%): These features contribute through XGBoost's native handling
  - Bureau/previous features: 0 is semantically correct (no prior history = zero loans)
  - EXT_SOURCE features: Missing counts captured in `EXT_SOURCE_MISSING_COUNT` feature
- **Simplicity**: No imputation pipeline to maintain in production
**Alternatives Considered:**
- Median/mode imputation: Loses missingness signal; introduces bias toward central tendency
- KNN imputation: Computationally expensive on 307K rows; creates data leakage risk
- MICE (Multiple Imputation by Chained Equations): Complex; overkill when model handles NaN natively
- Drop columns with >50% missing: Would remove potentially useful features (EXT_SOURCE_1 at 56% missing is highly predictive)
**Consequences:**
- No imputation step in the pipeline (simpler production code)
- Model learns different split behavior for present vs missing values
- EXT_SOURCE_1 (56% missing) remains usable as a feature
- `EXT_SOURCE_MISSING_COUNT` feature captures the count of unavailable external scores
- Production inference must pass NaN (not 0 or -1) for missing values to match training behavior
**Related:** `notebooks/01_eda.ipynb` (Section 4.2), `notebooks/02_FeatureEng.ipynb`

---

### [DECISION-014] Agent-Ready Model Card JSON + Risk Band Classification
**Date:** 2026-04-02
**Status:** Implemented
**Context:** The AI surveillance agent (NB05) and Streamlit dashboard consume model outputs, but rely on hardcoded values or text-based artifacts. A machine-readable model card enables programmatic access to all model metadata.
**Decision:** Export a comprehensive `models/model_card.json` containing model metadata, hyperparameters, performance metrics, thresholds, risk bands, top features, PSI baseline, calibration data, and known limitations. Define a 4-tier risk band system (Low/Medium/High/Critical) aligned with NB05 surveillance tiers.
**Rationale:**
- **Agent consumption**: JSON is directly parseable by the AI surveillance agent -- no regex or text scraping needed
- **Risk bands** provide consistent vocabulary across notebooks, dashboard, and agent:
  - Low (PD < 0.30): Standard monitoring
  - Medium (0.30-0.50): Enhanced monitoring
  - High (0.50-0.70): Active review required
  - Critical (>= 0.70): Immediate intervention
- **Self-documenting**: The JSON captures the full model specification in one file
- **Known limitations** explicitly flag EXT_SOURCE_WEIGHTED ~64% NaN rate and CODE_GENDER fair lending concern
- **Complements** the existing human-readable `reports/model_card.txt` (SR 11-7 format)
**Alternatives Considered:**
- YAML model card: Less universal than JSON for programmatic consumption
- Extend existing model_card.txt: Text format is not machine-parseable
- MLflow Model Registry: Requires infrastructure setup beyond project scope
- ONNX model metadata: Limited to model structure, not business context
**Consequences:**
- New artifact: `models/model_card.json` (~4-5 KB)
- Risk bands reusable across NB05, app.py, and future production scoring
- Must update model_card.json when model is retrained (add to retraining checklist)
**Related:** `notebooks/03_model_training_evaluation.ipynb` (Section 8.5), `notebooks/05_portfolio_surveillance.ipynb`

---

### [DECISION-015] PSI Baseline for Concept Drift Monitoring
**Date:** 2026-04-02
**Status:** Implemented
**Context:** The Streamlit dashboard (app.py) displays PSI values that are hardcoded, not computed from actual data. A real PSI baseline is needed for the AI surveillance agent to detect production drift.
**Decision:** Compute Population Stability Index (PSI) between train and test predicted probability distributions in NB03 as a baseline reference. Use 10 fixed equal-width bins on [0, 1] with epsilon smoothing.
**Rationale:**
- **Baseline establishment**: Train-vs-test PSI provides the "no drift" reference value; any production PSI significantly above this indicates real drift
- **Fixed equal-width bins** (not quantile): Standard for probability PSI in banking; directly comparable across time periods; quantile bins would create tiny intervals near 0 due to skewed PD distribution
- **Epsilon smoothing** (1e-6): Prevents log(0) errors in empty high-probability bins without materially affecting the PSI value
- **SR 11-7 compliance**: Demonstrates model stability awareness to regulators
- **Standard thresholds**: PSI < 0.10 (stable), 0.10-0.25 (monitor), >= 0.25 (recalibrate)
- **Dual visualization**: Overlay histogram + per-bin PSI contribution chart identifies exactly which probability ranges drive any observed shift
**Alternatives Considered:**
- Kolmogorov-Smirnov test: Less standard for credit risk drift monitoring
- Feature-level PSI (per column): More granular but higher complexity; suitable for production, not baseline
- Quantile-based bins: Would create unequal bin widths, harder to compare across time periods
- Chi-squared test: Requires minimum expected frequencies; PSI is more robust with small bins
**Consequences:**
- PSI value stored in `model_card.json` under `psi_baseline.train_vs_test`
- New visualization: `reports/psi_train_vs_test.png`
- app.py hardcoded PSI values should eventually be replaced with computed values (future task)
- Adds ~10 seconds to notebook runtime (predict_proba on full training set)
**Related:** `notebooks/03_model_training_evaluation.ipynb` (Section 8.3), `app.py` (Tab 3)

---

### [DECISION-016] Calibration-Enhanced Evaluation (Brier Score + Decile Table)
**Date:** 2026-04-02
**Status:** Implemented
**Context:** NB03 Section 8 had only a calibration curve plot -- no quantitative calibration metric (Brier Score) and no per-decile breakdown. These are essential for Basel III/IV Expected Loss calculations and regulatory validation.
**Decision:** Add Brier Score computation and a 10-decile calibration table (predicted PD vs actual default rate) to NB03 Section 8.
**Rationale:**
- **Brier Score** is the standard calibration metric for probability models (lower = better); enables comparison across model versions
- **Brier Skill Score** (1 - Brier/baseline) contextualizes performance relative to the naive predictor
- **Decile table** is the regulatory standard for calibration validation: regulators expect to see that predicted PDs align with observed default rates across the risk spectrum
- **Basel III/IV**: Expected Loss = PD x LGD x EAD; if PD is miscalibrated by decile, EL reserves are misallocated
- **IFRS 9**: ECL staging (1/2/3) depends on accurate PD thresholds; miscalibration causes incorrect stage assignments
- **Pred/Actual ratio** per decile: ratio close to 1.0 = well-calibrated; ratio > 1.0 = conservative (overestimates risk); ratio < 1.0 = dangerous (underestimates risk)
**Alternatives Considered:**
- Hosmer-Lemeshow test: Statistical test for calibration, but provides less interpretable output than a decile table
- Platt scaling / isotonic regression: Post-hoc recalibration; not needed if calibration is acceptable
- Expected Calibration Error (ECE): Used in deep learning; less standard in banking
**Consequences:**
- `brier_score` variable available for model_card.json (DECISION-014)
- `calibration_table` DataFrame shows per-decile predicted vs actual default rates
- Enables direct comparison with future model versions (Brier Score as benchmark)
**Related:** `notebooks/03_model_training_evaluation.ipynb` (Section 8.2)

---

### [DECISION-017] Data-Driven Business Cost Parameters (Real AMT_CREDIT Median)
**Date:** 2026-04-02
**Status:** Implemented
**Context:** Section 6 (Business Cost Analysis) used a hardcoded `LOAN_AMOUNT = 15000` which was an invented proxy. The actual dataset contains `AMT_CREDIT` with median $513,531 in the dataset's currency units.
**Decision:** Replace the hardcoded loan amount with `df['AMT_CREDIT'].median()` computed dynamically from the data. Add a currency disclaimer noting that '$' notation is used for readability but values are in the dataset's original currency units.
**Rationale:**
- **Data traceability**: Using the real median makes the analysis verifiable and defensible in interviews
- **Median over mean**: Median ($513,531) is more robust to outliers than mean ($599,026); the distribution has a heavy right tail (max $4,050,000)
- **Same optimal threshold**: The business-optimal threshold depends on the FN/FP cost ratio (LGD/profit_margin = 0.60/0.10 = 6:1), not absolute amounts -- the threshold is unchanged
- **Currency agnostic**: The cost-benefit framework works regardless of currency because conclusions depend on ratios, not absolute values
- **Interview readiness**: Shows the candidate uses real data rather than arbitrary assumptions
**Alternatives Considered:**
- Keep $15,000 with a disclaimer: Simpler but less credible
- Use mean instead of median: More sensitive to extreme outliers
- Calculate per-loan profit function (row-level AMT_CREDIT): More accurate but adds complexity for marginal insight; the portfolio-level analysis already demonstrates the framework
- Attempt to identify the actual currency: Not documented in the dataset; unnecessary since ratios drive the conclusions
**Consequences:**
- LOAN_AMOUNT now computed dynamically: `df['AMT_CREDIT'].median()`
- Dollar amounts in profit analysis are larger (~34x) but the 6:1 cost ratio and optimal threshold are preserved
- Currency disclaimer added as a markdown blockquote before Section 6
- DECISION-008 updated to reference data-driven values
**Related:** `notebooks/03_model_training_evaluation.ipynb` (Section 6), DECISION-008

---

### [DECISION-018] Bibliographic References as Inline Code Comments in NB03
**Date:** 2026-04-02
**Status:** Implemented
**Context:** Key technical methods in NB03 (Youden's J, profit-based classification, calibration, Brier Score, PSI) are grounded in academic literature. Adding citations directly in the code cells strengthens academic credibility and demonstrates domain awareness.
**Decision:** Add 2-4 line inline reference comments at the top of 5 code cells in NB03, citing the foundational papers for each technique. Comments placed inside code cells (not separate markdown cells) to keep them co-located with the implementation.
**References Added:**

| Cell | Section | Paper |
|------|---------|-------|
| 16 | 5.1 ROC Curve | Fluss, Faraggi & Reiser (2005). Biometrical Journal, 47(4), 458-472 |
| 23 | 6.2 Expected Profit | Verbraken, Bravo, Weber & Baesens (2014). EJOR, 238(2), 505-513 |
| 30 | 8.1 Calibration Curve | Bequé, Coussement, Gayler & Lessmann (2017). KBS, 134, 213-227 |
| 32 | 8.2 Brier Score | Bequé, Coussement, Gayler & Lessmann (2017). KBS, 134, 213-227 |
| 34 | 8.3 PSI | Yurdakul & Naranjo (2020). J. Risk Model Validation, 14(4), 89-100 |

**Rationale:**
- **Co-located with code**: References are immediately visible when reading the implementation, not buried in separate markdown cells that may be skipped
- **Interview readiness**: Demonstrates the candidate can trace implementation decisions to peer-reviewed research
- **Accuracy verified**: Each paper-to-code mapping was verified against the actual formulas (Youden's J = TPR - FPR, PSI = sum formula, Brier = MSE of probabilities)
- **Cell 23 nuance documented**: Comment explicitly notes the code is an "adapted profit-based framework" with fixed parameters, not the full EMP integration from Verbraken et al.
**Alternatives Considered:**
- Separate markdown cells per reference: Clutters the notebook structure; references are better as comments alongside the code they support
- References only in a bibliography at the end: Not visible when reading individual sections
- Footnotes in section headers: Would require modifying markdown cells and wouldn't be as directly tied to the code
**Consequences:**
- 5 code cells now have 4-5 additional comment lines each (no functional code changes)
- Helper script `src/add_references.py` preserved for reproducibility (one-shot use)
- References use standard academic citation format (Author (Year). "Title". Journal, Volume(Issue), Pages)
**Related:** `notebooks/03_model_training_evaluation.ipynb` (Cells 16, 23, 30, 32, 34)

---

### [DECISION-019] Post-Hoc Isotonic Calibration for Regulatory-Compliant PD Estimates
**Date:** 2026-04-02
**Status:** Implemented
**Context:** The calibration curve (Section 8.1) revealed that `scale_pos_weight = 11.39` inflates predicted probabilities 3-5x above observed default rates. The model is a good ranker (AUC 0.7793) but a poor probability estimator. This violates calibration requirements in FINMA Circular 2017/1, Basel IRB (CRR Art. 179), EU AI Act Art. 15, and IFRS 9 ECL staging.
**Decision:** Apply isotonic regression as a post-hoc calibration step. The calibrator is fitted on training predictions and applied to test predictions, overwriting `y_pred_proba` so all downstream cells (Brier Score, PSI, risk bands, model card) automatically use calibrated probabilities.
**Rationale:**
- **Preserves AUC**: Isotonic regression is monotonic — it remaps probabilities without changing the ranking, so AUC and Gini are unchanged
- **Academic backing**: Bequé et al. (2017), Table 4, recommends isotonic regression as the best-performing calibrator for credit scorecards (already cited in the notebook)
- **Regulatory compliance**: Calibrated PDs are required for Basel Expected Loss (PD x LGD x EAD), IFRS 9 staging thresholds, FINMA capital calculations, and nDSG right-to-explanation (customers must receive accurate risk assessments)
- **No retraining needed**: Post-hoc fix applied after model training — avoids the risk of degrading the XGBoost model
- **Production-ready**: Calibrator is saved as `models/calibrator.pkl` — one additional `calibrator.predict()` call in the inference pipeline
- **Before/after visualization**: Side-by-side calibration curves demonstrate the fix (saved to `reports/calibration_before_after.png`)
**Alternatives Considered:**
- Platt scaling (sigmoid): Assumes a logistic relationship between raw scores and true probabilities — too restrictive for tree-based models where the relationship is non-linear
- Remove `scale_pos_weight`: Would fix calibration but degrade recall on the minority class (defaults)
- Retrain with calibrated loss function: More complex, no guaranteed improvement over post-hoc calibration
- Use `CalibratedClassifierCV` with cross-validation: More principled but significantly more compute time; with 246K training rows, overfitting risk is negligible for isotonic regression
- Accept poor calibration with disclaimer: Not viable — regulatory requirements are explicit
**Consequences:**
- New artifact: `models/calibrator.pkl` (must be loaded alongside model in production)
- Business-optimal threshold will shift (now computed in calibrated probability space)
- All downstream metrics (Brier Score, PSI, risk bands, model card) reflect calibrated probabilities
- Downstream notebooks (NB04, NB05) and app.py must load and apply the calibrator when using raw model predictions — this is a future update task
- New visualization: `reports/calibration_before_after.png` (powerful interview talking point)
- PSI cell updated to use calibrated training predictions for a fair train-vs-test comparison
**Related:** `notebooks/03_model_training_evaluation.ipynb` (Section 8.1b), DECISION-008, Bequé et al. (2017)

---

### [DECISION-020] Expanded Regulatory Coverage in NB04 (FINMA, Swiss nDSG, EU AI Act)
**Date:** 2026-04-04
**Status:** Implemented
**Context:** NB04's regulatory references covered only US/EU frameworks (SR 11-7, Basel II/III, GDPR Art. 22, ECOA). The CREWS portfolio targets Swiss and international banking roles, and the project already references FINMA Circular 2017/1 and 2023/1 in NB03 (calibration decision). Consistency across notebooks and alignment with the target audience required extending the regulatory narrative throughout NB04.
**Decision:** Add EU AI Act Art. 10/13, FINMA Circular 2017/1, FINMA 2023/1, and Swiss nDSG to four locations in NB04: the intro cell, Section 7 header, Section 7.3 compliance table, and the executive summary regulatory readiness checklist. Also add a Notebook Scope section and a Technical References section to the intro cell.
**Rationale:**
- **FINMA 2017/1**: Swiss credit risk model governance requires independent validation and documentation — the model card (NB03) and SHAP analysis directly satisfy this
- **FINMA 2023/1**: Extends the 2017 framework to AI/ML models — requires auditability and explainability of automated decisions; SHAP provides exactly this
- **Swiss nDSG (2023)**: Grants data subjects the right to explanation for automated individual decisions (analogous to GDPR Art. 22); LIME/SHAP instance-level attributions constitute the required explanation record
- **EU AI Act Art. 10/13**: Credit scoring is a listed high-risk use case requiring technical documentation and human oversight — NB04's model card and adverse action template address these requirements
- **Portfolio coherence**: FINMA references already present in NB03 (DECISION-019) and app.py (Regulatory Compliance tab); NB04 must be consistent
- **Notebook Scope section**: Makes the connection between Kendall's τ (SR 11-7), LIME (cross-validation), and model_card.json (NB03) explicit in the intro — prevents reviewers from missing these design choices
- **Technical References**: Two papers (Ribeiro et al. 2016; Lin & Wang 2025) cited inline in code cells but not surfaced in the intro; the references section gives them visibility
**Alternatives Considered:**
- Add FINMA only to Section 7 (not the intro): Would leave the intro misaligned with the notebook's actual scope
- Create a dedicated Section 7.4 for Swiss/FINMA compliance: Over-structures a simple addition; bullet lists in existing sections are sufficient
- Add Lundberg & Lee (2017) as a third paper: Rejected — the two applied papers (LIME + SHAP stability) are more valuable than the foundational SHAP paper, which is universally assumed
**Consequences:**
- NB04 intro now matches the regulatory breadth of NB03 and app.py
- Section 7.3 table extended from 4 to 6 rows (adds FINMA 2017/1 and FINMA 2023/1/nDSG)
- Executive summary regulatory readiness checklist extended from 4 to 7 items
- Notebook Scope section clarifies the relationship between all four analytical threads (global SHAP, local SHAP, stability, LIME)
- Technical References section is the authoritative citation location; inline code comments remain as cross-references
**Related:** `notebooks/04_model_explainability.ipynb` (Cells 0, 46, 49, 53), DECISION-019, DECISION-009

---
