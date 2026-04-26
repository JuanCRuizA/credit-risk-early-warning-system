# Credit Risk Early Warning System (CREWS) - Issues & Solutions Log

## Purpose
Track technical problems encountered and how they were solved during the development of the CREWS project.

---

## Issue Template

### [ISSUE-XXX] Title
**Date:** YYYY-MM-DD
**Status:** Resolved | Investigating | Blocked
**Severity:** Critical | Medium | Low
**Problem:** What went wrong?
**Root Cause:** Why did it happen?
**Solution:** How was it fixed?
**Prevention:** How to avoid in future?

---

## Issues Log

### [ISSUE-001] Tool Decorator Syntax Incompatibility with Claude Agent SDK
**Date:** 2026-01-30
**Status:** Resolved
**Severity:** Critical
**Problem:** The AI agent in Notebook 05 failed to initialize tools. The `@tool` decorator from the Claude Agent SDK did not accept keyword arguments (`name=`, `description=`, `input_schema=`).
**Root Cause:** The Claude Agent SDK `@tool` decorator expects positional arguments, not keyword arguments. The initial implementation used `@tool(name="query_borrower_database", description="...", input_schema={...})` which is invalid syntax for the SDK.
**Solution:** Three iterative fix scripts were created in `src/`:
1. `fix_tool_decorators.py` (v1): Regex-based approach to identify and replace decorator patterns
2. `fix_tool_decorators_v2.py` (v2): Manual string replacement for each of 5 tools
3. `fix_tool_decorators_v3.py` (v3): Line-by-line processing with regex, successfully removed keyword arguments

Final working syntax: `@tool("tool_name", "Tool description")` with positional arguments only. Verified with `verify_tools.py`.
**Prevention:** Always test tool decorator syntax against the SDK documentation before building complex agent pipelines. The Claude Agent SDK uses a different decorator API than LangChain.

---

### [ISSUE-002] JSON Serialization Error with NumPy Types in Audit Logging
**Date:** 2026-01-30
**Status:** Resolved
**Severity:** Critical
**Problem:** `TypeError: Object of type float32 is not JSON serializable` when the AI agent attempted to log audit events to the audit trail. The `json.dumps()` call in the `log_audit_event()` function failed on NumPy data types.
**Root Cause:** XGBoost predictions return `numpy.float32` values, and pandas operations return `numpy.int64` values. Standard `json.dumps()` cannot serialize these NumPy-specific types.
**Solution:** Created `src/fix_json_serialization.py` which added `default=str` parameter to all `json.dumps()` calls in the `log_audit_event()` function. This converts any non-serializable types to their string representation.
**Prevention:** Always use `json.dumps(data, default=str)` when logging data that may contain NumPy or pandas types. Alternatively, convert NumPy types to Python native types before serialization.

---

### [ISSUE-003] XGBoost Dtype Incompatibility in Stress Testing
**Date:** 2026-01-30
**Status:** Resolved
**Severity:** Critical
**Problem:** `ValueError: DataFrame.dtypes for data must be int, float, bool or category` when running the `run_stress_test()` function in Notebook 05. XGBoost refused to accept the feature matrix for predictions.
**Root Cause:** After label encoding in Notebook 03, some categorical columns retained their `object` dtype when loaded from the SQLite database in Notebook 05. XGBoost requires numeric-compatible dtypes for all input features.
**Solution:** Created `src/fix_stress_test_dtype.py` (and v2) which added automatic dtype conversion before XGBoost predictions:
```python
for col in X_baseline.columns:
    if X_baseline[col].dtype == "object":
        X_baseline[col] = pd.Categorical(X_baseline[col]).codes
```
**Prevention:** Always verify feature dtypes after loading from databases or CSV files. Add dtype validation as a pre-prediction check in all inference pipelines.

---

### [ISSUE-004] DAYS_EMPLOYED Anomalous Value (365243)
**Date:** 2026-01-25
**Status:** Resolved
**Severity:** Medium
**Problem:** The `DAYS_EMPLOYED` column contained the value 365243 (~1,000 years) for a significant portion of records, which is clearly not a valid employment duration and distorted statistical analysis.
**Root Cause:** This is a known data encoding in the Home Credit dataset where `DAYS_EMPLOYED = 365243` represents unemployed or retired applicants. It affects approximately 55,374 records (18% of the dataset).
**Solution:** Detected and documented in Notebook 01 (EDA). The `EMPLOYMENT_YEARS` derived feature sets these values to `NaN`:
```python
df.loc[df['DAYS_EMPLOYED'] == 365243, 'EMPLOYMENT_YEARS'] = np.nan
```
In Notebook 02 (Feature Engineering), a binary flag `FLAG_UNEMPLOYED` was created to capture this signal explicitly.
**Prevention:** Always check for sentinel values in imported datasets. The Home Credit column description file documents this encoding, but it's easy to miss.

---

### [ISSUE-005] Class Imbalance Affecting Model Recall
**Date:** 2026-01-25
**Status:** Resolved
**Severity:** Medium
**Problem:** Initial XGBoost model with default parameters showed poor recall on the minority class (defaulters). The model was biased toward predicting "no default" due to the 92%/8% class distribution.
**Root Cause:** With only 8.07% default rate (class imbalance ratio of 11.4:1), the model's default loss function penalizes false negatives equally as false positives, leading to poor minority class detection.
**Solution:** Applied `scale_pos_weight = 11.39` (ratio of non-defaults to defaults) in the XGBoost configuration. This weights the positive class (defaults) 11.39x more heavily during training, forcing the model to pay more attention to correctly identifying defaulters.
**Prevention:** Always check class distribution before training. For imbalanced datasets, use `scale_pos_weight` (XGBoost), `class_weight='balanced'` (sklearn), or sampling techniques (SMOTE).

---

### [ISSUE-006] Fair Lending Compliance Check Failure
**Date:** 2026-01-30
**Status:** Investigating
**Severity:** Medium
**Problem:** The automated fair lending compliance check in Notebook 05 returned status `FAIL` with message "Disparate Impact Analysis Requires Review". The verification subagent flagged the term "age" in the model's report.
**Root Cause:** The model uses `CODE_GENDER` and `AGE_YEARS` (derived from `DAYS_BIRTH`) as features. SHAP analysis showed `CODE_GENDER` has a mean SHAP contribution of -0.0154, and `AGE_YEARS` shows a correlation with default risk. Both are protected attributes under ECOA/Regulation B.
**Solution:** Documented in the Model Card and SHAP explainability analysis. Recommendations generated:
1. Conduct regular disparate impact analysis by protected class
2. Monitor approval rates by gender and age groups
3. Consider removing or constraining sensitive features
4. Implement second-look programs for borderline rejections

**Prevention:** Include fair lending impact assessment as a mandatory step before model deployment. Establish threshold criteria for acceptable disparate impact ratios (e.g., 80% rule).

---

### [ISSUE-007] Division by Zero in Feature Engineering Ratios
**Date:** 2026-01-25
**Status:** Resolved
**Severity:** Low
**Problem:** Several engineered ratio features (e.g., `DEBT_TO_INCOME`, `CREDIT_TO_GOODS`, `ANNUITY_TO_CREDIT`) produced `inf` or `NaN` values when the denominator was zero.
**Root Cause:** Some loan applications have zero values for `AMT_INCOME_TOTAL`, `AMT_GOODS_PRICE`, or `AMT_CREDIT` in edge cases, causing division by zero in ratio calculations.
**Solution:** Infinite values were replaced with `NaN` after ratio computation in Notebook 02. XGBoost handles `NaN` values natively by learning optimal split directions, so no imputation was needed.
**Prevention:** Always add post-computation validation for ratio features. Consider using `np.divide(a, b, where=b!=0)` or explicit zero-checks before division.

---

### [ISSUE-008] Multiple Agent Session Restarts in Notebook 05
**Date:** 2026-01-30
**Status:** Resolved
**Severity:** Low
**Problem:** The audit trail log shows 8 separate `SESSION_START` events on 2026-01-30, indicating the AI agent was restarted multiple times before completing a full analysis cycle.
**Root Cause:** Early sessions failed due to the tool decorator issue (ISSUE-001), JSON serialization errors (ISSUE-002), and stress test dtype issues (ISSUE-003). Each fix required restarting the agent session.
**Solution:** After resolving all three underlying issues (ISSUE-001 through ISSUE-003), the agent completed a full 4-phase analysis cycle successfully. The final session (session_id: `20260131_065202`) ran all phases without errors.
**Prevention:** Implement unit tests for each tool function before running the full agent pipeline. Test tools independently to isolate failures.

---

### [ISSUE-009] Spanish Text Remaining in Notebook 03
**Date:** 2026-02-01
**Status:** Resolved
**Severity:** Low
**Problem:** Notebook 03 (Model Training & Evaluation) contained comments and print statements in Spanish, inconsistent with the rest of the project which is in English.
**Root Cause:** The notebook was initially developed with Spanish comments (e.g., "Definir ruta base", "Cargar feature names", "GENERAR PREDICCIONES") and the translation to English was not completed during the initial development phase.
**Solution:** All Spanish text in two cells was translated to English, including header comments, step labels, print statements, and the author attribution.
**Prevention:** Establish a single language (English) for all code comments and documentation at the start of the project.

---

### [ISSUE-010] Missing EDA Visualizations and Null Value Documentation
**Date:** 2026-02-15
**Status:** Resolved
**Severity:** Low
**Problem:** Notebook 01 (EDA) lacked key analytical visualizations (correlation heatmap, box plots for outliers, missing value patterns) and had no formal documentation of the null handling strategy.
**Root Cause:** The initial EDA focused on descriptive statistics and pivot tables but did not include analytical visualizations for multicollinearity detection, outlier quantification, or missing value pattern analysis.
**Solution:** Enhanced Notebook 01 with the following additions:
- Section 5.1: Correlation heatmap + multicollinearity check
- Section 4.1: Missing value matrix + missingness correlation heatmap
- Section 4.2: Formal null handling strategy (categorized by missing percentage threshold)
- Section 5.2: Box plots + IQR-based outlier quantification
- Improved DAYS_EMPLOYED anomaly documentation
- Removed all emojis for professional tone

**Prevention:** Use a standard EDA checklist covering: target analysis, missing values (pattern + strategy), correlations, outliers, and categorical analysis.

---

### [ISSUE-011] LIME Feature Name Extraction from Rule Strings
**Date:** 2026-03-28
**Status:** Resolved
**Severity:** Low
**Problem:** LIME's `as_list()` method returns feature rules as strings (e.g., "0.32 < EXT_SOURCE_MEAN <= 0.55") rather than clean feature names. This makes direct comparison with SHAP feature names non-trivial.
**Root Cause:** LIME discretizes continuous features and generates rule-based explanations. The output format includes threshold values embedded in the feature name string.
**Solution:** Implemented string matching against `X_sample.columns` to extract the original feature name from each LIME rule string. This handles all feature name formats in the dataset.
**Prevention:** When using LIME for comparison with other methods, always implement a feature name extraction step. Consider using `lime_exp.as_map()` as an alternative that returns feature indices instead of rule strings.

---

### [ISSUE-013] Notebook JSON Corruption During Programmatic Cell Insertion
**Date:** 2026-04-02
**Status:** Resolved
**Severity:** Medium
**Problem:** When inserting 8 new cells into NB03 via a Python script, the initial `json.dump()` with `ensure_ascii=False` raised a `UnicodeEncodeError: surrogates not allowed`, partially writing the file and corrupting the notebook JSON.
**Root Cause:** The original notebook contained non-BMP Unicode characters (emojis like `📊`, `💡`, `🔑` in print statements). Combined with `ensure_ascii=False` on Windows, these caused a surrogate encoding error during JSON serialization. The partial write left the file in an invalid JSON state.
**Solution:** Two-part fix:
1. Restored the original notebook from git: `git checkout -- notebooks/03_model_training_evaluation.ipynb`
2. Changed JSON serialization to `ensure_ascii=True` (escapes all non-ASCII as `\uXXXX` sequences) and added safe-write pattern (write to `.tmp` file first, verify valid JSON, then move to final path)
**Prevention:** Always use safe-write pattern for notebook modifications: write to temp file, validate JSON, then atomically replace. Use `ensure_ascii=True` when notebook may contain non-BMP Unicode characters.

---

### [ISSUE-014] Truncated Threshold Search Range Produced False Business Optimum
**Date:** 2026-04-02
**Status:** Resolved
**Severity:** Medium
**Problem:** The business-optimal threshold was reported as 0.59, but the profit-vs-threshold graph was cut off at the red dashed line — it was impossible to see whether profit continued rising beyond that point. Visual inspection suggested the trend was still increasing.
**Root Cause:** The threshold search range in Cell 23 was `np.arange(0.05, 0.60, 0.02)`, which stops at 0.58. The profit curve was still ascending at the right edge, so `idxmax()` simply returned the last evaluated threshold (0.59) as the "optimal" — a truncation artifact, not a true peak.
**Solution:** Extended the search range to `np.arange(0.05, 0.96, 0.02)`, revealing the true profit peak at **0.79**. Also improved the visualization (Cell 24):
- Added green/red fill to distinguish positive vs negative profit zones
- Added red star marker at the peak profit point
- Added horizontal zero line for break-even reference
- Extended x-axis to 0.95 to show the full decline after the peak
**Prevention:** Always verify that optimization search ranges extend well beyond the expected optimum. Visually confirm that the objective function shows a clear peak followed by decline — a graph that ends at the maximum is a red flag for truncation.

---

### [ISSUE-015] pandas `applymap` Deprecation in Dashboard PSI Table
**Date:** 2026-04-11
**Status:** Resolved
**Severity:** Medium
**Problem:** `AttributeError: DataFrame.style.applymap()` crashed the PSI table cell-coloring code in the Streamlit dashboard.
**Root Cause:** `applymap` was deprecated in pandas 2.1 and removed. The equivalent element-wise styling method is now `.map()`.
**Solution:** Changed `psi_data.style.applymap(color_psi)` → `psi_data.style.map(color_psi)` in `app.py`.
**Prevention:** When upgrading pandas, check for `applymap` usage in any styling code. The method was renamed to `.map()` in pandas 2.1.

---

### [ISSUE-016] UnicodeDecodeError on Windows for Text Files Containing Emoji
**Date:** 2026-04-11
**Status:** Resolved
**Severity:** Critical
**Problem:** All five Streamlit dashboard tabs crashed with `UnicodeDecodeError: 'charmap' codec can't decode byte 0x90`. The error surfaced when loading `executive_summary.txt` (which contains emoji like 📊, 💼 and box-drawing characters).
**Root Cause:** Windows uses the `cp1252` codec as the default file encoding. `cp1252` cannot handle emoji or box-drawing characters. Any `open()` call without an explicit `encoding` argument falls back to the system default on Windows.
**Solution:** Added `encoding='utf-8'` to all four `open()` calls in `app.py`: `load_model_card()`, `load_agent_output()`, the `executive_summary.txt` read block, and the `adverse_action_notice_sample.txt` read block.
**Prevention:** Always specify `encoding='utf-8'` explicitly on any `open()` call that reads files containing non-ASCII characters. This is especially important for cross-platform projects where files are authored on Linux/Mac but served on Windows.

---

### [ISSUE-017] Risk Calculator KeyError on Interaction Features
**Date:** 2026-04-11
**Status:** Resolved
**Severity:** Medium
**Problem:** Clicking "Calculate Risk" in the sidebar raised `KeyError: "['EXT_SCORE_x_PAYMENT_BURDEN', 'EXT_SCORE_x_AGE', 'EXT_SCORE_x_DEBT_RATIO', 'HAS_BUREAU_HISTORY', 'HAS_PREV_APPLICATION'] not in index"`.
**Root Cause:** `population_medians.csv` only contains features present in the raw dataset and simple engineered features. The five interaction and flag features (`EXT_SCORE_x_*`, `HAS_BUREAU_HISTORY`, `HAS_PREV_APPLICATION`) are computed during feature engineering in NB02 but are not included in the medians file. The risk calculator tried to build `X_calc` by direct indexing `feat_vals[feature_names]`, which fails when keys are missing.
**Solution:** Two-part fix in `app.py`:
1. Explicitly compute the five missing features from already-set values before building `X_calc`: `EXT_SCORE_x_PAYMENT_BURDEN = EXT_SOURCE_MEAN × PAYMENT_BURDEN`, `EXT_SCORE_x_AGE = EXT_SOURCE_MEAN × calc_age`, `EXT_SCORE_x_DEBT_RATIO = EXT_SOURCE_MEAN × DEBT_TO_INCOME`, `HAS_BUREAU_HISTORY = 1`, `HAS_PREV_APPLICATION = 1`.
2. Changed `feat_vals[feature_names].values` → `feat_vals.reindex(feature_names, fill_value=0).values` so any remaining gap features default to 0 instead of raising a `KeyError`.
**Prevention:** When the risk calculator uses `population_medians.csv` as a base, any feature that isn't in that file must be explicitly computed. Use `.reindex(fill_value=0)` as a defensive pattern for future feature additions.

---

### [ISSUE-018] Git Local Config Overriding Global — Contributions Not Credited to Liverpool Account
**Date:** 2026-04-11
**Status:** Resolved
**Severity:** Medium
**Problem:** GitHub contributions grid showed no activity since 2026-03-23 despite active commits. All recent commits were attributed to `fantastic112172@gmail.com` (a blocked account), not the Liverpool university email linked to the GitHub profile.
**Root Cause:** An old local repository config (`git config --local`) was setting `user.email = fantastic112172@gmail.com` inside the CREWS repo. Git's config precedence (local > global > system) means the local config silently overrode the correct global config (`j.ruiz-arteaga@liverpool.ac.uk`).
**Solution:** 
1. Verified with `git config --local --list` that the local override existed.
2. Removed it: `git config --local --unset user.email` and `git config --local --unset user.name`.
3. Amended the stale README commit: `git commit --amend --author="Juan Carlos Ruiz Arteaga <j.ruiz-arteaga@liverpool.ac.uk>" --no-edit`.
4. Pushed — confirmed 1 contribution on 2026-04-11 in the GitHub grid.
**Prevention:** After changing GitHub account email, always verify local repo configs with `git config --local --list`. Local overrides are repo-specific and persist independently of global settings.

---

### [ISSUE-019] app.py Cost Model Inconsistent with NB03 Profit-Maximization Framework
**Date:** 2026-04-26
**Status:** Resolved
**Severity:** Medium
**Problem:** The Streamlit dashboard showed threshold 0.51 (statistical optimum) with higher "Net Savings" than threshold 0.79 (business optimum), directly contradicting the profit-maximization analysis in NB03. At 0.51 the app displayed $875M vs $279M at 0.79, making the business threshold look economically inferior.
**Root Cause:** Two compounding errors introduced when app.py was built independently of NB03:
1. Wrong parameters: `LGD=0.45` (should be 0.60), `FP_COST=50.0` flat review fee (should be `AVG_LOAN * 0.10`, the foregone loan profit), and `AVG_LOAN` computed with `.mean()` (should be `.median()` per NB03 §6.1).
2. Wrong formula: `net_savings = total_defaults * FN_COST - fn * FN_COST - fp * FP_COST` counts only defaults caught minus false alarm costs, omitting the profit from correctly approved good customers (TN). NB03's formula is `net_profit = tn * FP_COST - fn * FN_COST - fp * FP_COST`. With a flat $50 FP cost, the omission was invisible; with the correct $51K FP cost it is decisive (54,811 good loans correctly approved at 0.79 vs 42,860 at 0.51 = $612M difference).
**Solution:** Four changes to `app.py` (cost block and downstream displays):
1. `AVG_LOAN`: `.mean()` → `.median()`
2. `LGD`: `0.45` → `0.60`
3. `FP_COST`: `50.0` → `AVG_LOAN * PROFIT_RATE` (PROFIT_RATE=0.10), yielding ~$51,206 (foregone loan profit per rejected good customer)
4. Formula + label: `net_savings = no_model_cost - total_cost` → `net_profit = tn * FP_COST - fn * FN_COST - fp * FP_COST`; KPI tile renamed "Expected Net Profit"; delta shows incremental value over no-model baseline (+$141M at 0.79).
Downstream labels updated consistently: "False Alarm Cost" → "Foregone Loan Profit" in Cost Analysis panel; "Total Cost" → "Expected Net Profit" in Cost-Benefit table; sidebar cost structure note updated.
Validation: FN/FP ratio = 6.0x; Expected Net Profit at 0.79 = $1.511B; at 0.51 = $970M. 0.79 wins by $541M.
**Prevention:** When deploying a dashboard from a notebook, trace all cost parameters directly to the notebook source. Cross-check by computing the metric at both the statistical and business threshold and confirming the business threshold dominates. See DECISION-027.

---

### [ISSUE-012] EMPLOYMENT_YEARS Inconsistency Between NB01 and NB02
**Date:** 2026-03-29
**Status:** Resolved
**Severity:** Low
**Problem:** Notebook 01 (EDA) set `EMPLOYMENT_YEARS` to `NaN` for the DAYS_EMPLOYED = 365243 sentinel (unemployed/retired), but Notebook 02 (Feature Engineering) set it to `0`. This inconsistency could surface during an interview review.
**Root Cause:** The two notebooks were developed at different times. NB01 treated unemployed as "unknown employment duration" (NaN), while NB02 treated it as "zero years of employment" (0). Both are defensible, but the discrepancy means the EDA analysis and feature engineering do not align.
**Solution:** Changed NB02 to use `np.nan` (matching NB01). XGBoost handles NaN natively by learning optimal split directions, so the signal is preserved. The `FLAG_UNEMPLOYED` binary feature (created on the next line) explicitly captures the unemployed/retired signal.
**Prevention:** When converting EDA findings into feature engineering, verify that the handling logic matches. Document sentinel value treatment in a single location and reference it from both notebooks.

---
