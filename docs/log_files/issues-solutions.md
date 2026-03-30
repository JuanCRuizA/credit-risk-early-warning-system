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

### [ISSUE-012] EMPLOYMENT_YEARS Inconsistency Between NB01 and NB02
**Date:** 2026-03-29
**Status:** Resolved
**Severity:** Low
**Problem:** Notebook 01 (EDA) set `EMPLOYMENT_YEARS` to `NaN` for the DAYS_EMPLOYED = 365243 sentinel (unemployed/retired), but Notebook 02 (Feature Engineering) set it to `0`. This inconsistency could surface during an interview review.
**Root Cause:** The two notebooks were developed at different times. NB01 treated unemployed as "unknown employment duration" (NaN), while NB02 treated it as "zero years of employment" (0). Both are defensible, but the discrepancy means the EDA analysis and feature engineering do not align.
**Solution:** Changed NB02 to use `np.nan` (matching NB01). XGBoost handles NaN natively by learning optimal split directions, so the signal is preserved. The `FLAG_UNEMPLOYED` binary feature (created on the next line) explicitly captures the unemployed/retired signal.
**Prevention:** When converting EDA findings into feature engineering, verify that the handling logic matches. Document sentinel value treatment in a single location and reference it from both notebooks.

---
