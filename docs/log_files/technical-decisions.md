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
**Date:** 2026-01-25
**Status:** Implemented
**Context:** The default threshold (0.5) is arbitrary. Credit risk models need thresholds optimized for both statistical performance and business value.
**Decision:** Calculate two optimal thresholds:
- **Statistical optimal (Youden's J):** 0.5092 -- maximizes sensitivity + specificity
- **Business optimal:** 0.5900 -- maximizes expected profit using cost parameters
**Rationale:**
- **Two separate concerns**: Statistical performance and business value often conflict
- **Youden's J** provides the mathematically optimal balance of sensitivity/specificity
- **Profit optimization** incorporates real business costs:
  - Average loan: $15,000
  - Loss Given Default (LGD): 60%
  - Cost of default: $9,000
  - Profit per good loan: $1,500
- **Business threshold** produces $36.8M maximum expected profit
- Both thresholds stored in `models/thresholds.pkl` for deployment flexibility
**Alternatives Considered:**
- Fixed threshold (0.5): Arbitrary, not optimized for either statistical or business criteria
- Single threshold (statistical only): Ignores business economics
- Multi-threshold strategy (like BAFS auto-block/manual review/auto-approve): Considered but simpler binary decision is more appropriate for credit approval (approve/reject)
- Cost-sensitive learning (custom loss function): More complex, harder to validate
**Consequences:**
- At business threshold (0.59): Precision 0.2272, Recall 0.5531
- Maximum profit: $36,819,000
- Higher threshold (0.59 vs 0.50) means fewer rejections but some defaults slip through
- Decision shifts from "catch all defaults" to "maximize portfolio profitability"
**Related:** `notebooks/03_model_training_evaluation.ipynb` (Sections 5-6)

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
