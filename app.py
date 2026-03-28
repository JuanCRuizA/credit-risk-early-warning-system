"""
CREWS - Credit Risk Early Warning System
Streamlit Dashboard - Interactive Portfolio Analytics & Regulatory Compliance

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from PIL import Image
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, auc, precision_score, recall_score
)
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# Page Configuration
# =====================================================================
st.set_page_config(
    page_title="CREWS - Credit Risk Early Warning System",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# Custom CSS - Professional Banking Aesthetic
# =====================================================================
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1a365d; }
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.85rem;
        padding: 20px 0;
        border-top: 1px solid #dee2e6;
        margin-top: 40px;
    }
    .footer a { color: #1a365d; text-decoration: none; }
    .footer a:hover { text-decoration: underline; }
    .toc-box {
        background: #eef2f7;
        padding: 14px 20px;
        border-radius: 8px;
        border-left: 4px solid #1a365d;
        margin-bottom: 16px;
    }
    .toc-box a { color: #1a365d; text-decoration: none; }
    .toc-box a:hover { text-decoration: underline; }
    h1 > a, h2 > a, h3 > a, h4 > a, h5 > a, h6 > a,
    .stMarkdown h1 > a, .stMarkdown h2 > a, .stMarkdown h3 > a {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Paths
# =====================================================================
try:
    BASE_PATH = Path(__file__).parent.resolve()
except NameError:
    BASE_PATH = Path.cwd()

MODEL_PATH = BASE_PATH / 'models'
REPORTS_PATH = BASE_PATH / 'reports'
DOCS_PATH = BASE_PATH / 'docs' / 'screenshots'


# =====================================================================
# Data & Model Loading (cached)
# =====================================================================
@st.cache_resource
def load_model_artifacts():
    """Load XGBoost model, thresholds, and feature names."""
    model = joblib.load(MODEL_PATH / 'xgb_credit_model.pkl')
    thresholds = joblib.load(MODEL_PATH / 'thresholds.pkl')
    feature_names = joblib.load(MODEL_PATH / 'feature_names.pkl')
    return model, thresholds, feature_names


@st.cache_resource
def load_risk_calculator_deps():
    """Load label encoders and medians for the risk calculator."""
    le = joblib.load(MODEL_PATH / 'label_encoders_v2.pkl')
    modes = joblib.load(MODEL_PATH / 'categorical_modes.pkl')
    medians = pd.read_csv(MODEL_PATH / 'population_medians.csv', index_col=0).squeeze()
    return le, modes, medians


@st.cache_data
def load_test_data():
    """Load the pre-computed test set with predictions."""
    return pd.read_csv(BASE_PATH / 'test_dashboard.csv')


# Load everything
try:
    model, thresholds, feature_names = load_model_artifacts()
    df_test = load_test_data()
    BUSINESS_THRESHOLD = thresholds.get('business_optimal', 0.59)
    STAT_THRESHOLD = thresholds.get('statistical_optimal', 0.509)

    y_test = df_test['TARGET'].values
    fraud_scores = df_test['predicted_pd'].values

    # Cost assumptions for credit risk
    AVG_LOAN = df_test['AMT_CREDIT'].mean() if 'AMT_CREDIT' in df_test.columns else 500000
    LGD = 0.45  # Loss Given Default (Basel IRB foundation, unsecured consumer)
    FN_COST = AVG_LOAN * LGD  # Cost of missing a default
    FP_COST = 50.0  # Manual review cost

except Exception as e:
    st.error(f"Failed to load model or data: {e}")
    st.info("Ensure model artifacts exist in models/ and test_dashboard.csv at project root.")
    st.stop()


# =====================================================================
# Feature Labels (human-readable)
# =====================================================================
FEATURE_LABELS = {
    'EXT_SOURCE_MEAN': 'External Score (Combined)',
    'EXT_SOURCE_MAX': 'External Score (Best)',
    'EXT_SOURCE_MIN': 'External Score (Lowest)',
    'EXT_SOURCE_3': 'External Score 3',
    'EXT_SOURCE_1': 'External Score 1',
    'EXT_SOURCE_WEIGHTED': 'External Score (Weighted)',
    'ANNUITY_TO_CREDIT': 'Annuity-to-Credit Ratio',
    'CREDIT_TO_GOODS': 'Credit-to-Goods Ratio',
    'CODE_GENDER': 'Gender',
    'AMT_ANNUITY': 'Monthly Annuity ($)',
    'AMT_CREDIT': 'Loan Amount ($)',
    'AMT_GOODS_PRICE': 'Goods Price ($)',
    'AMT_INCOME_TOTAL': 'Annual Income ($)',
    'AGE_YEARS': 'Age (Years)',
    'EMPLOYMENT_YEARS': 'Employment (Years)',
    'PAYMENT_BURDEN': 'Payment Burden (%)',
    'DEBT_TO_INCOME': 'Debt-to-Income Ratio',
    'PREV_REFUSAL_RATE': 'Previous Refusal Rate',
    'BUREAU_DEBT_TO_CREDIT_RATIO': 'Bureau Debt-to-Credit',
    'OWN_CAR_AGE': 'Car Age (Years)',
    'NAME_EDUCATION_TYPE': 'Education Level',
}


# =====================================================================
# Reusable Footer
# =====================================================================
FOOTER_HTML = """
<div class="footer">
    <strong>CREWS - Credit Risk Early Warning System</strong><br>
    <a href="https://github.com/JuanCRuizA/credit-risk-early-warning-system"
       target="_blank">
        https://github.com/JuanCRuizA/credit-risk-early-warning-system
    </a><br>
    Developed by Juan Carlos Ruiz Arteaga | Banking Data Scientist<br>
    MSc in Data Science &amp; AI, University of Liverpool<br>
    Contact: j.ruiz-arteaga@liverpool.ac.uk
</div>
"""


def render_footer():
    st.markdown("---")
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


# =====================================================================
# Sidebar - Global Configuration
# =====================================================================
with st.sidebar:
    st.title("CREWS")
    st.caption("Credit Risk Early Warning System")
    st.markdown("---")

    st.subheader("Global Filters")

    risk_threshold = st.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(round(BUSINESS_THRESHOLD, 2)),
        step=0.01,
        help=(
            f"Loans with PD above this threshold are flagged as high-risk. "
            f"Business optimal: {BUSINESS_THRESHOLD:.3f} | "
            f"Statistical optimal: {STAT_THRESHOLD:.3f}"
        ),
    )

    st.markdown("---")

    # Export
    st.subheader("Export")
    flagged_mask = fraud_scores >= risk_threshold
    export_df = df_test[['SK_ID_CURR', 'TARGET', 'predicted_pd']].copy()
    if 'AMT_CREDIT' in df_test.columns:
        export_df['AMT_CREDIT'] = df_test['AMT_CREDIT']
    export_df = export_df[flagged_mask]
    csv_data = export_df.to_csv(index=False)
    st.download_button(
        label=f"Export Flagged ({flagged_mask.sum():,} loans)",
        data=csv_data,
        file_name="flagged_high_risk_loans.csv",
        mime="text/csv",
        help="Download all loans above the current threshold as CSV.",
    )

    st.markdown("---")

    # Risk Calculator
    st.subheader("Individual Risk Calculator")
    st.caption("Score a single loan application")

    calc_ext = st.slider("External Credit Score", 0.0, 1.0, 0.5, 0.01, key="calc_ext")
    calc_age = st.number_input("Age (years)", 20, 80, 35, key="calc_age")
    calc_credit = st.number_input("Loan Amount ($)", 10000, 2000000, 200000, 5000, key="calc_credit")
    calc_annuity = st.number_input("Monthly Payment ($)", 500, 50000, 5000, 100, key="calc_annuity")

    if st.button("Calculate Risk", type="primary"):
        try:
            le_dict, cat_modes, pop_medians = load_risk_calculator_deps()

            # Build feature vector from population medians
            feat_vals = pop_medians.copy()

            # Override with user inputs
            feat_vals['EXT_SOURCE_3'] = calc_ext
            feat_vals['EXT_SOURCE_2'] = calc_ext
            feat_vals['EXT_SOURCE_1'] = calc_ext * 0.9
            feat_vals['EXT_SOURCE_MEAN'] = calc_ext
            feat_vals['EXT_SOURCE_WEIGHTED'] = calc_ext
            feat_vals['EXT_SOURCE_MAX'] = calc_ext
            feat_vals['EXT_SOURCE_MIN'] = calc_ext * 0.9
            feat_vals['EXT_SOURCE_PRODUCT'] = calc_ext ** 3

            feat_vals['DAYS_BIRTH'] = -calc_age * 365.25
            feat_vals['AGE_YEARS'] = calc_age

            feat_vals['AMT_CREDIT'] = calc_credit
            feat_vals['AMT_GOODS_PRICE'] = calc_credit * 0.9

            feat_vals['AMT_ANNUITY'] = calc_annuity
            income = pop_medians.get('AMT_INCOME_TOTAL', 150000)
            feat_vals['ANNUITY_TO_CREDIT'] = calc_annuity / calc_credit if calc_credit > 0 else 0
            feat_vals['CREDIT_TO_GOODS'] = calc_credit / (calc_credit * 0.9) if calc_credit > 0 else 1
            feat_vals['PAYMENT_BURDEN'] = (calc_annuity * 12) / income if income > 0 else 0
            feat_vals['DEBT_TO_INCOME'] = calc_credit / income if income > 0 else 0

            # Encode categoricals with population modes
            for col in cat_modes:
                if col in le_dict:
                    mode_val = cat_modes[col]
                    if mode_val in le_dict[col].classes_:
                        feat_vals[col] = le_dict[col].transform([mode_val])[0]
                    else:
                        feat_vals[col] = 0

            # Build DataFrame for prediction
            X_calc = pd.DataFrame([feat_vals[feature_names].values], columns=feature_names)
            X_calc = X_calc.replace([np.inf, -np.inf], [10, -10]).fillna(0)
            risk_score = model.predict_proba(X_calc)[:, 1][0]

            st.markdown("### Risk Assessment")
            if risk_score < 0.10:
                st.success(f"**Low Risk:** {risk_score:.1%}")
            elif risk_score < risk_threshold:
                st.warning(f"**Moderate Risk:** {risk_score:.1%}")
            else:
                st.error(f"**High Risk:** {risk_score:.1%}")

            ecl = calc_credit * risk_score * LGD
            st.info(f"Expected Credit Loss: ${ecl:,.0f}")

        except Exception as e:
            st.error(f"Calculation error: {e}")

    st.markdown("---")

    with st.expander("About & Methods"):
        st.markdown(
            "- **Dataset:** Home Credit Default Risk (307,511 loans)\n"
            "- **Model:** XGBoost with 211 engineered features\n"
            f"- **Cost structure:** ${FN_COST:,.0f} FN / ${FP_COST:.0f} FP\n"
            "- **Explainability:** SHAP TreeExplainer + LIME\n"
            "- **Compliance:** SR 11-7, Basel III/IV, IFRS 9, ECOA, EU AI Act, FINMA, nDSG"
        )


# =====================================================================
# Apply Threshold
# =====================================================================
y_pred = (fraud_scores >= risk_threshold).astype(int)


# =====================================================================
# MAIN TABS
# =====================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Portfolio Overview",
    "Model Performance",
    "AI Agent Insights",
    "SHAP Explainability",
    "Regulatory Compliance",
])


# =====================================================================
# TAB 1 - Portfolio Overview
# =====================================================================
with tab1:
    st.header("Portfolio Overview")
    st.caption("Key performance indicators and IFRS 9 staging for the credit portfolio")

    # Confusion matrix components
    tp = int(((y_test == 1) & (y_pred == 1)).sum())
    fp = int(((y_test == 0) & (y_pred == 1)).sum())
    fn = int(((y_test == 1) & (y_pred == 0)).sum())
    tn = int(((y_test == 0) & (y_pred == 0)).sum())
    total_defaults = int(y_test.sum())
    total_loans = len(y_test)
    recall = tp / total_defaults if total_defaults > 0 else 0
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0

    missed_cost = fn * FN_COST
    review_cost = fp * FP_COST
    total_cost = missed_cost + review_cost
    no_model_cost = total_defaults * FN_COST
    net_savings = no_model_cost - total_cost

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Loans (Test Set)", f"{total_loans:,}", f"{total_defaults:,} defaults ({y_test.mean():.1%})")
    k2.metric("High-Risk Flagged", f"{y_pred.sum():,}", f"{y_pred.sum()/total_loans:.1%} of portfolio")
    k3.metric("Defaults Detected", f"{tp:,} / {total_defaults:,}", f"{recall:.1%} recall")
    k4.metric("Net Savings vs No Model", f"${net_savings:,.0f}", f"${no_model_cost:,.0f} baseline")

    st.markdown("---")

    # IFRS 9 Staging
    st.subheader("IFRS 9 Expected Credit Loss Staging")
    st.caption(
        "Portfolio segmented by credit quality. Stage 1: 12-month ECL. "
        "Stage 2: Lifetime ECL (significant increase in credit risk). "
        "Stage 3: Lifetime ECL (credit-impaired)."
    )

    # Staging thresholds
    stage1_mask = fraud_scores < 0.10
    stage2_mask = (fraud_scores >= 0.10) & (fraud_scores < risk_threshold)
    stage3_mask = fraud_scores >= risk_threshold

    amt_col = 'AMT_CREDIT' if 'AMT_CREDIT' in df_test.columns else None

    def compute_stage_ecl(mask, lifetime=False):
        """Compute ECL for a stage."""
        pds = fraud_scores[mask]
        if amt_col:
            eads = df_test[amt_col].values[mask]
        else:
            eads = np.full(mask.sum(), AVG_LOAN)
        if lifetime:
            # Lifetime ECL ~ PD * LGD * EAD (simplified: multiply by maturity factor ~3)
            ecl = (pds * LGD * eads * 3).sum()
        else:
            # 12-month ECL
            ecl = (pds * LGD * eads).sum()
        return len(pds), eads.sum(), ecl

    s1_count, s1_exposure, s1_ecl = compute_stage_ecl(stage1_mask, lifetime=False)
    s2_count, s2_exposure, s2_ecl = compute_stage_ecl(stage2_mask, lifetime=True)
    s3_count, s3_exposure, s3_ecl = compute_stage_ecl(stage3_mask, lifetime=True)
    total_ecl = s1_ecl + s2_ecl + s3_ecl
    total_exposure = s1_exposure + s2_exposure + s3_exposure

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown(
            "<div style='background:#2e7d32;color:white;padding:12px;border-radius:8px;text-align:center'>"
            "<strong>Stage 1 - Performing</strong><br>"
            f"PD < 10% | 12-month ECL<br><br>"
            f"<span style='font-size:1.8rem'>{s1_count:,}</span> loans<br>"
            f"Exposure: ${s1_exposure:,.0f}<br>"
            f"ECL Provision: ${s1_ecl:,.0f}"
            "</div>",
            unsafe_allow_html=True,
        )
    with sc2:
        st.markdown(
            "<div style='background:#e65100;color:white;padding:12px;border-radius:8px;text-align:center'>"
            "<strong>Stage 2 - Underperforming</strong><br>"
            f"10% ≤ PD < {risk_threshold:.0%} | Lifetime ECL<br><br>"
            f"<span style='font-size:1.8rem'>{s2_count:,}</span> loans<br>"
            f"Exposure: ${s2_exposure:,.0f}<br>"
            f"ECL Provision: ${s2_ecl:,.0f}"
            "</div>",
            unsafe_allow_html=True,
        )
    with sc3:
        st.markdown(
            "<div style='background:#c62828;color:white;padding:12px;border-radius:8px;text-align:center'>"
            "<strong>Stage 3 - Non-Performing</strong><br>"
            f"PD ≥ {risk_threshold:.0%} | Credit-Impaired<br><br>"
            f"<span style='font-size:1.8rem'>{s3_count:,}</span> loans<br>"
            f"Exposure: ${s3_exposure:,.0f}<br>"
            f"ECL Provision: ${s3_ecl:,.0f}"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"**Total ECL Provision: ${total_ecl:,.0f}** | "
                f"Coverage Ratio: {total_ecl/total_exposure:.2%} of total exposure")

    # Staging distribution bar
    stage_df = pd.DataFrame({
        'Stage': ['Stage 1\nPerforming', 'Stage 2\nUnderperforming', 'Stage 3\nNon-Performing'],
        'Loans': [s1_count, s2_count, s3_count],
        'ECL ($M)': [s1_ecl / 1e6, s2_ecl / 1e6, s3_ecl / 1e6],
    })
    fig_stage = px.bar(
        stage_df, x='Stage', y='Loans', color='Stage',
        color_discrete_map={
            'Stage 1\nPerforming': '#2e7d32',
            'Stage 2\nUnderperforming': '#e65100',
            'Stage 3\nNon-Performing': '#c62828',
        },
        text='Loans',
    )
    fig_stage.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig_stage.update_layout(showlegend=False, height=350, title="IFRS 9 Stage Distribution")
    st.plotly_chart(fig_stage, use_container_width=True)

    st.markdown("---")

    # Risk Score Distribution
    left, right = st.columns(2)
    with left:
        st.subheader("PD Score Distribution")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=fraud_scores[y_test == 0], nbinsx=50, name='Non-Default',
            marker_color='#2196F3', opacity=0.6,
        ))
        fig_hist.add_trace(go.Histogram(
            x=fraud_scores[y_test == 1], nbinsx=50, name='Default',
            marker_color='#f44336', opacity=0.6,
        ))
        fig_hist.add_vline(x=risk_threshold, line_dash='dash', line_color='#333',
                           annotation_text=f'Threshold ({risk_threshold:.2f})')
        fig_hist.update_layout(
            barmode='overlay', height=400,
            xaxis_title='Predicted PD', yaxis_title='Count',
            title='Distribution of Default Probabilities',
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with right:
        st.subheader("Cost Analysis")
        c1, c2 = st.columns(2)
        c1.metric("Missed Default Cost", f"${missed_cost:,.0f}", f"{fn:,} missed x ${FN_COST:,.0f}")
        c2.metric("False Alarm Cost", f"${review_cost:,.0f}", f"{fp:,} reviews x ${FP_COST:.0f}")

        st.markdown("")
        perf_df = pd.DataFrame({
            'Metric': ['Recall (Default Detection)', 'Precision', 'F1-Score',
                       'False Positive Rate', 'Threshold Applied'],
            'Value': [
                f"{recall:.2%}",
                f"{precision_val:.2%}",
                f"{2*precision_val*recall/(precision_val+recall):.4f}" if (precision_val+recall) > 0 else "0",
                f"{fp/(fp+tn):.2%}" if (fp+tn) > 0 else "0%",
                f"{risk_threshold:.3f}",
            ],
        })
        st.table(perf_df)

    render_footer()


# =====================================================================
# TAB 2 - Model Performance
# =====================================================================
with tab2:
    st.header("Model Performance")
    st.caption("XGBoost model evaluation on the held-out test set (61,503 loans)")

    # Live Confusion Matrix + ROC
    st.subheader("Performance at Current Threshold")
    cm = confusion_matrix(y_test, y_pred)
    tn_v, fp_v, fn_v, tp_v = cm.ravel()

    cm_left, cm_right = st.columns(2)

    with cm_left:
        # Confusion matrix with cost overlay
        cm_labels = [
            [f"TN<br>{tn_v:,}<br>$0", f"FP<br>{fp_v:,}<br>${fp_v * FP_COST:,.0f}"],
            [f"FN<br>{fn_v:,}<br>${fn_v * FN_COST:,.0f}", f"TP<br>{tp_v:,}<br>Prevented"],
        ]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, colorscale='Blues', showscale=False,
            text=[[cm_labels[0][0], cm_labels[0][1]], [cm_labels[1][0], cm_labels[1][1]]],
            texttemplate="%{text}",
            textfont=dict(size=13),
            x=['Predicted Non-Default', 'Predicted Default'],
            y=['Actual Non-Default', 'Actual Default'],
        ))
        fig_cm.update_layout(
            title=f'Confusion Matrix (threshold = {risk_threshold:.3f})',
            height=400, xaxis_title='Predicted', yaxis_title='Actual',
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with cm_right:
        # Live ROC curve
        fpr_arr, tpr_arr, _ = roc_curve(y_test, fraud_scores)
        roc_auc_val = roc_auc_score(y_test, fraud_scores)
        tpr_op = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0
        fpr_op = fp_v / (fp_v + tn_v) if (fp_v + tn_v) > 0 else 0

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr_arr, y=tpr_arr, mode='lines', name=f'XGBoost (AUC={roc_auc_val:.4f})',
            line=dict(color='#1565C0', width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Random',
            line=dict(color='gray', dash='dash'),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[fpr_op], y=[tpr_op], mode='markers', name=f'Operating Point ({risk_threshold:.2f})',
            marker=dict(color='red', size=12, symbol='x'),
        ))
        fig_roc.update_layout(
            title='ROC Curve', height=400,
            xaxis_title='False Positive Rate', yaxis_title='True Positive Rate (Recall)',
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # Pre-computed images
    st.subheader("Model Evaluation (from Notebook 03)")
    img_l, img_r = st.columns(2)
    with img_l:
        roc_img = REPORTS_PATH / 'precision_recall_curve.png'
        if roc_img.exists():
            st.image(Image.open(roc_img), caption='Precision-Recall Curve', use_container_width=True)
    with img_r:
        cost_img = REPORTS_PATH / 'profit_vs_threshold.png'
        if cost_img.exists():
            st.image(Image.open(cost_img), caption='Profit vs Threshold', use_container_width=True)

    img_l2, img_r2 = st.columns(2)
    with img_l2:
        cal_img = REPORTS_PATH / 'calibration_curve.png'
        if cal_img.exists():
            st.image(Image.open(cal_img), caption='Calibration Curve', use_container_width=True)
    with img_r2:
        thresh_img = REPORTS_PATH / 'threshold_analysis.png'
        if thresh_img.exists():
            st.image(Image.open(thresh_img), caption='Threshold Analysis', use_container_width=True)

    st.markdown("---")

    # Feature Importance
    st.subheader("Feature Importance (SHAP)")
    shap_imp = REPORTS_PATH / 'shap_global_importance.png'
    if shap_imp.exists():
        st.image(Image.open(shap_imp), use_container_width=True)
    else:
        fi_img = REPORTS_PATH / 'feature_importance.png'
        if fi_img.exists():
            st.image(Image.open(fi_img), use_container_width=True)

    st.markdown("---")

    # Cost-Benefit Table
    st.subheader("Cost-Benefit Analysis by Threshold")
    thresholds_sweep = sorted(set([
        0.10, 0.20, 0.30, 0.40,
        round(float(STAT_THRESHOLD), 2),
        round(float(BUSINESS_THRESHOLD), 2),
        0.70, 0.80, 0.90,
    ]))
    rows = []
    for t in thresholds_sweep:
        yp = (fraud_scores >= t).astype(int)
        t_tp = int(((y_test == 1) & (yp == 1)).sum())
        t_fp = int(((y_test == 0) & (yp == 1)).sum())
        t_fn = int(((y_test == 1) & (yp == 0)).sum())
        t_rec = t_tp / total_defaults if total_defaults > 0 else 0
        t_pre = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
        t_cost = t_fn * FN_COST + t_fp * FP_COST
        marker = " *" if abs(t - risk_threshold) < 0.005 else ""
        rows.append({
            'Threshold': f"{t:.2f}{marker}",
            'Recall': f"{t_rec:.1%}",
            'Precision': f"{t_pre:.1%}",
            'Defaults Caught': f"{t_tp:,}",
            'False Positives': f"{t_fp:,}",
            'Missed Defaults': f"{t_fn:,}",
            'Total Cost': f"${t_cost:,.0f}",
        })
    st.caption("* = current threshold")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    render_footer()


# =====================================================================
# TAB 3 - AI Agent Insights
# =====================================================================
with tab3:
    st.header("AI Agent Portfolio Surveillance")
    st.caption(
        "Autonomous AI agent (Claude Sonnet 4) performs 4-phase hierarchical analysis "
        "of the credit portfolio with regulatory compliance checks."
    )

    st.info("**Last Analysis:** January 31, 2026 | **Status:** Completed | **Model:** Claude Sonnet 4")

    # Agent screenshot
    agent_img = DOCS_PATH / 'ai_agent_overview.png'
    if agent_img.exists():
        st.image(Image.open(agent_img), caption='AI Agent Portfolio Surveillance Output',
                 use_container_width=True)

    st.markdown("---")

    st.subheader("Key Findings & Recommendations")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""
### Phase A: Data Validation
- **307,511 loans** analyzed
- **Distribution drift** detected in:
    - `EXT_SOURCE_3` (credit bureau score)
    - `DAYS_BIRTH` (age distribution)
    - `AMT_CREDIT` (loan amounts)
- Model operating outside calibration zone

### Phase B: Risk Identification
- **42,073 borrowers** flagged as high-risk (PD > 0.59)
- Represents **13.68% of total portfolio**
""")

    with col_r:
        st.markdown("""
### Phase C: Trend Analysis
- **Performance deterioration** over loan vintage:
    - New loans (0-6 mo): 42.7% high-risk
    - Medium loans (6-18 mo): 60.3% high-risk
    - Mature loans (18+ mo): 63.8% high-risk
- Underwriting quality decline post-2022

### Phase D: Recommendations
1. **Suspend origination** until risk addressed
2. **Intensive collection** for 42,073 high-risk borrowers
3. **Underwriting audit** for recent vintages
4. **Model recalibration** -- detected PSI drift
""")

    st.markdown("---")

    # PSI Drift Monitoring
    st.subheader("Population Stability Index (PSI) -- Drift Monitoring")
    st.caption(
        "PSI measures distribution shift between training and current data. "
        "< 0.10: Stable | 0.10-0.25: Investigate | > 0.25: Significant shift."
    )

    psi_data = pd.DataFrame([
        {'Feature': 'EXT_SOURCE_3', 'PSI': 0.28, 'Status': 'Significant Shift',
         'Action': 'Model recalibration recommended'},
        {'Feature': 'DAYS_BIRTH', 'PSI': 0.15, 'Status': 'Moderate Shift',
         'Action': 'Monitor -- age distribution changing'},
        {'Feature': 'AMT_CREDIT', 'PSI': 0.12, 'Status': 'Moderate Shift',
         'Action': 'Monitor -- loan amounts trending higher'},
        {'Feature': 'EXT_SOURCE_MEAN', 'PSI': 0.08, 'Status': 'Stable',
         'Action': 'No action required'},
        {'Feature': 'AMT_ANNUITY', 'PSI': 0.05, 'Status': 'Stable',
         'Action': 'No action required'},
    ])

    def color_psi(val):
        if val >= 0.25:
            return 'background-color: #ffcdd2'
        elif val >= 0.10:
            return 'background-color: #fff9c4'
        return 'background-color: #c8e6c9'

    st.dataframe(
        psi_data.style.applymap(color_psi, subset=['PSI']),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")

    # Regulatory alignment
    st.subheader("Regulatory Compliance Status")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.success("**SR 11-7 Alignment**\n- Ongoing monitoring\n- Model validation\n- Audit trail")
    with rc2:
        st.success("**Basel III/IV**\n- Risk-weighted assets\n- Capital adequacy\n- Stress testing")
    with rc3:
        st.success("**IFRS 9**\n- ECL provisioning\n- Forward-looking\n- Stage classification")

    render_footer()


# =====================================================================
# TAB 4 - SHAP Explainability
# =====================================================================
with tab4:
    st.header("SHAP Explainability")
    st.caption(
        "Model explanations using SHAP TreeExplainer. "
        "Global plots show overall feature importance; "
        "local waterfall plots explain individual loan decisions."
    )

    # Global SHAP
    st.subheader("Global Feature Importance")
    g1, g2 = st.columns(2)
    with g1:
        shap_bar = REPORTS_PATH / 'shap_global_importance.png'
        if shap_bar.exists():
            st.image(Image.open(shap_bar), caption='Mean |SHAP| Feature Importance',
                     use_container_width=True)
    with g2:
        shap_bee = REPORTS_PATH / 'shap_summary_beeswarm.png'
        if shap_bee.exists():
            st.image(Image.open(shap_bee), caption='SHAP Summary (Beeswarm)',
                     use_container_width=True)

    st.markdown("---")

    # Dependence Plots
    st.subheader("SHAP Dependence Plots")
    st.caption("How each feature's value affects the model's prediction.")

    dep_files = [
        ('shap_dependence_ext_source_2.png', 'External Score 2'),
        ('shap_dependence_age_years.png', 'Age'),
        ('shap_dependence_amt_credit.png', 'Loan Amount'),
        ('shap_dependence_amt_income.png', 'Income'),
        ('shap_dependence_employment.png', 'Employment Duration'),
    ]
    dep_cols = st.columns(3)
    for i, (fname, label) in enumerate(dep_files):
        fpath = REPORTS_PATH / fname
        if fpath.exists():
            with dep_cols[i % 3]:
                st.image(Image.open(fpath), caption=label, use_container_width=True)

    st.markdown("---")

    # Case Studies
    st.subheader("Case Study Explorer")
    st.caption(
        "Representative loan decisions covering all confusion matrix quadrants. "
        "Each case shows the SHAP waterfall and a plain-English explanation."
    )

    cases = {
        "Case 1: True Positive -- High-Risk Default Correctly Flagged": {
            "score": 0.85, "actual": "DEFAULT", "decision": "FLAGGED",
            "waterfall_file": "shap_waterfall_true_positive.png",
            "features": {
                "External Score (Combined)": "0.12 (very low)",
                "Annuity-to-Credit Ratio": "0.068 (high burden)",
                "Employment": "2.1 years (short tenure)",
                "Age": "28 years (younger borrower)",
                "Loan Amount": "$320,000",
                "Previous Refusal Rate": "40%",
            },
            "explanation": (
                "This loan was correctly identified as high-risk. The dominant "
                "driver was the very low external credit score (SHAP pushing strongly "
                "toward default), indicating poor credit history from bureau data. "
                "Short employment tenure and young age reinforced the risk signal. "
                "The high previous application refusal rate (40%) suggests a pattern "
                "of credit difficulties. The model correctly flagged this borrower, "
                "who subsequently defaulted."
            ),
            "drivers_title": "Key Risk Drivers",
            "drivers": [
                "External credit score 0.12 -- far below population median of 0.51",
                "Previous refusal rate 40% -- history of rejected applications",
                "Employment 2.1 years -- below stability threshold",
                "Age 28 -- younger borrowers carry statistically higher risk",
            ],
        },
        "Case 2: False Positive -- Good Borrower Incorrectly Flagged": {
            "score": 0.62, "actual": "NON-DEFAULT", "decision": "FLAGGED",
            "waterfall_file": "shap_waterfall_false_positive.png",
            "features": {
                "External Score (Combined)": "0.35 (below average)",
                "Annuity-to-Credit Ratio": "0.055",
                "Employment": "8.5 years (stable)",
                "Age": "42 years",
                "Loan Amount": "$450,000 (large)",
                "Previous Refusal Rate": "0%",
            },
            "explanation": (
                "This borrower was incorrectly flagged as high-risk despite "
                "repaying the loan successfully. The below-average external score "
                "(0.35) was the primary false signal -- while below the population "
                "median, this borrower had stable employment (8.5 years) and zero "
                "previous refusals. The large loan amount ($450,000) contributed "
                "to the elevated score. This case illustrates a model limitation: "
                "external scores dominate the prediction, occasionally overriding "
                "positive behavioral signals."
            ),
            "drivers_title": "Key Factors in False Alert",
            "drivers": [
                "External score 0.35 -- below average but not critically low",
                "Large loan amount $450K -- size alone increases perceived risk",
                "Stable employment 8.5 years -- positive signal underweighted",
            ],
        },
        "Case 3: False Negative -- Missed Default": {
            "score": 0.38, "actual": "DEFAULT", "decision": "APPROVED",
            "waterfall_file": "shap_waterfall_false_negative.png",
            "features": {
                "External Score (Combined)": "0.55 (above average)",
                "Annuity-to-Credit Ratio": "0.042 (moderate)",
                "Employment": "5.3 years",
                "Age": "45 years",
                "Loan Amount": "$180,000",
                "Previous Refusal Rate": "0%",
            },
            "explanation": (
                "The model failed to detect this default because all features "
                "appeared healthy. The external credit score (0.55) was above "
                "average, employment was stable, and the borrower had no previous "
                "refusals. This represents a fundamental model limitation: the "
                "default was driven by factors not captured in the feature set "
                "(e.g., sudden income loss, health event, or macroeconomic shock). "
                "This case motivates incorporating macroeconomic indicators and "
                "real-time behavioral data in future model versions."
            ),
            "drivers_title": "Key Factors in Missed Detection",
            "drivers": [
                "External score 0.55 -- above average, suppressed risk signal",
                "Zero previous refusals -- clean credit history masked emerging risk",
                "Moderate loan amount -- within normal range, no size-based signal",
            ],
            "improvement": (
                "Incorporate forward-looking macroeconomic indicators (unemployment "
                "rate, GDP growth, interest rate changes) and real-time behavioral "
                "signals (payment delays, utilization spikes) to detect defaults "
                "driven by external shocks rather than static borrower characteristics."
            ),
        },
    }

    selected = st.selectbox("Select a case study:", list(cases.keys()))
    case = cases[selected]

    st.markdown("---")

    m1, m2, m3 = st.columns(3)
    m1.metric("Default Probability", f"{case['score']:.4f}")
    m2.metric("Model Decision", case['decision'])
    m3.metric("Actual Outcome", case['actual'])

    st.subheader("Borrower Features")
    feat_items = list(case['features'].items())
    mid = (len(feat_items) + 1) // 2
    fc1, fc2 = st.columns(2)
    with fc1:
        for k, v in feat_items[:mid]:
            st.markdown(f"**{k}:** {v}")
    with fc2:
        for k, v in feat_items[mid:]:
            st.markdown(f"**{k}:** {v}")

    st.subheader("SHAP Waterfall -- Why the Model Made This Decision")
    wf_path = REPORTS_PATH / case['waterfall_file']
    if wf_path.exists():
        st.image(Image.open(wf_path), caption=f"SHAP waterfall: {selected.split(' --')[0]}",
                 use_container_width=True)
    else:
        st.info(f"Waterfall plot ({case['waterfall_file']}) not found. Run notebook 04.")

    st.subheader("Model Decision Explanation")
    st.write(case['explanation'])

    st.subheader(case.get('drivers_title', 'Key Risk Drivers'))
    for d in case['drivers']:
        st.markdown(f"- {d}")

    if 'improvement' in case:
        st.subheader("Recommended Improvement")
        st.write(case['improvement'])

    render_footer()


# =====================================================================
# TAB 5 - Regulatory Compliance
# =====================================================================
with tab5:
    st.header("Regulatory Compliance", anchor="compliance-top")
    st.caption("Model governance, fair lending review, IFRS 9 methodology, and audit readiness")

    st.markdown("""
<div class="toc-box">
<strong>Table of Contents</strong><br><br>
&nbsp;&nbsp;1. <a href="#sr-11-7-checklist">Production Readiness Checklist</a><br>
&nbsp;&nbsp;2. <a href="#fair-lending-review">Fair Lending Considerations</a><br>
&nbsp;&nbsp;3. <a href="#model-governance">Model Governance Framework</a><br>
&nbsp;&nbsp;4. <a href="#ifrs-9-methodology">IFRS 9 ECL Methodology</a><br>
&nbsp;&nbsp;5. <a href="#right-to-explanation">Right-to-Explanation Capabilities</a><br>
&nbsp;&nbsp;6. <a href="#basel-alignment">Basel III/IV Alignment</a><br>
&nbsp;&nbsp;7. <a href="#eu-ai-act">EU AI Act (2024) Compliance</a><br>
&nbsp;&nbsp;8. <a href="#finma-circular">FINMA Circular 2017/1 (Switzerland)</a><br>
&nbsp;&nbsp;9. <a href="#swiss-ndsg">Swiss nDSG Data Protection</a>
</div>
""", unsafe_allow_html=True)

    # Section 1: SR 11-7
    st.subheader("1. Production Readiness Checklist", anchor="sr-11-7-checklist")

    done_items = [
        "Model documentation (purpose, inputs, outputs, assumptions)",
        "Performance metrics on held-out test data (AUC, Gini, KS)",
        "Global explainability (SHAP feature importance, beeswarm)",
        "Local explainability (individual loan SHAP waterfalls)",
        "Limitations and known risks documented",
        "Fair lending feature review conducted",
        "Business threshold optimization (profit maximization)",
        "Audit trail requirements specified (Notebook 05)",
    ]
    pending_items = [
        "Disparate impact testing (requires demographic data)",
        "Champion/challenger framework (multi-model comparison)",
        "Ongoing monitoring dashboard (PSI drift detection)",
        "Quarterly model revalidation schedule",
        "Out-of-time validation (temporal holdout)",
        "Independent model validation (second-line review)",
    ]

    chk1, chk2 = st.columns(2)
    with chk1:
        st.markdown("**Completed**")
        for item in done_items:
            st.checkbox(item, value=True, disabled=True, key=f"done_{item[:30]}")
    with chk2:
        st.markdown("**Pending**")
        for item in pending_items:
            st.checkbox(item, value=False, disabled=True, key=f"pend_{item[:30]}")

    st.markdown("[Back to top](#compliance-top)")
    st.markdown("---")

    # Section 2: Fair Lending
    st.subheader("2. Fair Lending Considerations", anchor="fair-lending-review")

    fl_data = pd.DataFrame([
        {'Feature': 'External Credit Scores (EXT_SOURCE_1/2/3)',
         'Risk Level': 'LOW',
         'Assessment': 'Objective credit bureau data. No direct demographic proxy. Monitor for disparate impact across segments.'},
        {'Feature': 'CODE_GENDER',
         'Risk Level': 'HIGH',
         'Assessment': 'Direct protected attribute (ECOA). Present in model features. Requires disparate impact testing and justification.'},
        {'Feature': 'AGE_YEARS / DAYS_BIRTH',
         'Risk Level': 'MEDIUM',
         'Assessment': 'SHAP shows younger borrowers carry higher risk. Age is protected under ECOA. Monitor approval rates by age cohort.'},
        {'Feature': 'NAME_FAMILY_STATUS',
         'Risk Level': 'MEDIUM',
         'Assessment': 'Marital status is protected under ECOA. Monitor for disparate treatment of single vs married applicants.'},
        {'Feature': 'NAME_EDUCATION_TYPE',
         'Risk Level': 'LOW-MEDIUM',
         'Assessment': 'Education correlates with income. May be proxy for socioeconomic status. Monitor across educational groups.'},
    ])
    st.table(fl_data)
    st.markdown(
        "**Overall Assessment:** CODE_GENDER is a direct protected attribute and requires "
        "disparate impact analysis. AGE_YEARS and NAME_FAMILY_STATUS are ECOA-protected. "
        "Full disparate impact testing pending availability of demographic outcome data."
    )

    st.markdown("[Back to top](#compliance-top)")
    st.markdown("---")

    # Section 3: Model Governance
    st.subheader("3. Model Governance Framework", anchor="model-governance")

    gov1, gov2 = st.columns(2)
    with gov1:
        st.markdown("**Model Identification**")
        st.text("Name:     CREWS (XGBoost Credit Risk Model)")
        st.text("Version:  1.0")
        st.text("Type:     Gradient Boosted Decision Tree")
        st.text("Purpose:  Credit default probability estimation")
        st.text("Features: 211 engineered from 7 data sources")
        st.text("Date:     January 2026")
        st.markdown("")
        st.markdown("**Monitoring Schedule**")
        sched = pd.DataFrame([
            {'Frequency': 'Daily', 'Activity': 'Alert volume, auto-flag count, queue size'},
            {'Frequency': 'Weekly', 'Activity': 'AUC, Gini, recall by risk tier'},
            {'Frequency': 'Monthly', 'Activity': 'PSI drift analysis, feature stability, IFRS 9 stage migration'},
            {'Frequency': 'Quarterly', 'Activity': 'Full revalidation, threshold recalibration, stress testing'},
            {'Frequency': 'Annual', 'Activity': 'Comprehensive SR 11-7 review, ECOA/fair lending audit'},
        ])
        st.dataframe(sched, use_container_width=True, hide_index=True)

    with gov2:
        st.markdown("**Model Risk Classification**")
        st.text("Recommended Tier:  Tier 2 (Material)")
        st.text("Rationale:         Material financial impact")
        st.text("Review Cycle:      Quarterly")
        st.markdown("")
        st.markdown("**Key Assumptions**")
        st.markdown(
            "1. Training default patterns represent future defaults\n"
            "2. Stratified 80/20 split preserves class distribution\n"
            "3. Client identity: SK_ID_CURR (unique application ID)\n"
            "4. LGD assumption: 45% (Basel IRB foundation)\n"
            "5. External scores available at application time\n"
            "6. scale_pos_weight = 11.39 for class imbalance"
        )

    st.markdown("[Back to top](#compliance-top)")
    st.markdown("---")

    # Section 4: IFRS 9 Methodology
    st.subheader("4. IFRS 9 ECL Methodology", anchor="ifrs-9-methodology")

    st.markdown("""
**Expected Credit Loss Formula:**

`ECL = PD x LGD x EAD`

where:
- **PD** (Probability of Default): XGBoost model output, calibrated on 307,511 historical loans
- **LGD** (Loss Given Default): 45% -- Basel IRB foundation approach for unsecured consumer lending
- **EAD** (Exposure at Default): Outstanding loan amount (AMT_CREDIT)

**Staging Criteria:**

| Stage | Condition | ECL Horizon | Trigger |
|-------|-----------|-------------|---------|
| Stage 1 -- Performing | PD < 10% | 12-month ECL | Initial recognition, no SICR |
| Stage 2 -- Underperforming | 10% <= PD < business threshold | Lifetime ECL | Significant Increase in Credit Risk (SICR) |
| Stage 3 -- Non-Performing | PD >= business threshold | Lifetime ECL (impaired) | Credit-impaired, objective evidence of loss |

**Significant Increase in Credit Risk (SICR) Definition:**
- PD increases from below 10% at origination to above 10% at reporting date
- Quantitative: relative PD increase > 100% (doubling) AND absolute increase > 5pp
- Qualitative: 30+ days past due (backstop per IFRS 9 B5.5.22)

**Lifetime ECL Simplification:**
- Lifetime ECL approximated as PD x LGD x EAD x maturity factor (3 years average remaining term)
- This is a simplified approach; production systems use marginal PD term structures

**Forward-Looking Adjustments:**
- Base scenario (60% weight): current macroeconomic conditions
- Optimistic scenario (20% weight): GDP growth +1%, unemployment -1pp
- Pessimistic scenario (20% weight): GDP growth -2%, unemployment +2pp
- Scenario weighting follows IFRS 9 B5.5.42 requirements
""")

    st.markdown("[Back to top](#compliance-top)")
    st.markdown("---")

    # Section 5: Right to Explanation
    st.subheader("5. Right-to-Explanation Capabilities", anchor="right-to-explanation")

    st.markdown("""
Applicants whose loans are denied or flagged for review may request an explanation.
SHAP values provide a complete, auditable explanation at the individual loan level.

**Applicable Frameworks:**
- **ECOA / Reg B** -- US: adverse action notices must cite specific reasons for denial
- **GDPR Art. 22** -- EU: right to explanation for automated decisions with legal effects
- **IFRS 9** -- Staging decisions must be explainable and auditable

**For any loan application, the system can generate:**
1. **Feature-level attribution** -- which factors contributed to the decision
2. **Quantified contribution** -- how much each factor affected the PD score
3. **Comparison to baseline** -- score relative to average default probability (8.07%)

**Adverse Action Notice Generation:**
1. Model scores the application (PD output)
2. SHAP values computed for the application
3. Top 4 negative factors extracted from SHAP
4. Plain-English reasons generated (e.g., "Credit history length below lender requirements")
5. Reasons provided to applicant per ECOA Reg B requirements

**Audit Trail Requirements:**
- SHAP values stored at scoring time for every decision
- Retention: minimum 7 years (US regulatory standard)
- Log fields: application_id, pd_score, threshold, decision, top_shap_factors, model_version, timestamp
""")

    st.markdown("[Back to top](#compliance-top)")
    st.markdown("---")

    # Section 6: Basel Alignment
    st.subheader("6. Basel III/IV Alignment", anchor="basel-alignment")

    st.markdown("""
**Risk-Weighted Assets (RWA) Framework:**

Under the Basel IRB approach, risk weights are derived from PD, LGD, and EAD:

| Parameter | CREWS Value | Source |
|-----------|-------------|--------|
| PD (portfolio average) | 8.07% | XGBoost model output |
| LGD | 45% | Basel IRB foundation (unsecured) |
| EAD | AMT_CREDIT per loan | Application data |
| Maturity (M) | 2.5 years | Basel default assumption |
| Correlation (R) | 0.15 | Basel retail portfolio formula |

**Capital Requirements:**
- Minimum CET1: 4.5% of RWA
- Capital conservation buffer: 2.5%
- Countercyclical buffer: 0-2.5% (jurisdiction-dependent)
- Total minimum: 7.0-9.5% of RWA

**Stress Testing Reference:**
- Notebook 05 implements three stress scenarios:
  - Interest rate shock (+200 bps)
  - Income reduction (-20%)
  - Employment stress scenario
- Results feed into capital adequacy assessment

**Model Use in Basel Framework:**
- CREWS PD estimates can serve as PIT (Point-in-Time) inputs to IRB calculations
- For TTC (Through-the-Cycle) PD required by some jurisdictions, apply long-run average adjustment
- Model performance (AUC 0.779, Gini 0.559) meets minimum discrimination requirements for IRB approval
""")

    st.markdown("[Back to top](#compliance-top)")
    st.markdown("---")

    # Section 7: EU AI Act (2024)
    st.subheader("7. EU AI Act (2024) Compliance", anchor="eu-ai-act")

    st.markdown("""
**Classification:** CREWS is a **high-risk AI system** under the EU AI Act.

**Legal Basis:** Regulation (EU) 2024/1689, Annex III, Section 5(b):
> *"AI systems intended to be used to evaluate the creditworthiness of natural persons
> or establish their credit score"* are classified as high-risk.

This classification triggers mandatory requirements under Title III, Chapter 2 (Articles 8-15).
""")

    eu_ai_data = pd.DataFrame([
        {'Requirement': 'Art. 9 -- Risk Management System',
         'Status': 'PARTIAL',
         'CREWS Implementation': 'SHAP+LIME dual explainability, business threshold optimization, stress testing in Notebook 05. Residual risk: model drift monitoring is designed but not yet automated in production.'},
        {'Requirement': 'Art. 10 -- Data Governance',
         'Status': 'COMPLIANT',
         'CREWS Implementation': 'Home Credit dataset with documented provenance. EDA (Notebook 01) covers data quality, missing value strategy, outlier analysis. Feature engineering documented in Notebook 02.'},
        {'Requirement': 'Art. 11 -- Technical Documentation',
         'Status': 'COMPLIANT',
         'CREWS Implementation': 'Model card (reports/model_card.txt), technical decisions log, 5 notebooks with full methodology. SHAP feature importance archived.'},
        {'Requirement': 'Art. 12 -- Record-Keeping',
         'Status': 'COMPLIANT',
         'CREWS Implementation': 'Audit trail logging (audit_trail.log) with session IDs, timestamps, and action summaries. SHAP values stored per decision.'},
        {'Requirement': 'Art. 13 -- Transparency',
         'Status': 'COMPLIANT',
         'CREWS Implementation': 'SHAP waterfall plots for individual decisions. LIME provides independent model-agnostic validation. Adverse action notice generation capability.'},
        {'Requirement': 'Art. 14 -- Human Oversight',
         'Status': 'PARTIAL',
         'CREWS Implementation': 'Dashboard enables human review of flagged loans. Business threshold allows manual override. Full automation not implemented (human-in-the-loop by design).'},
        {'Requirement': 'Art. 15 -- Accuracy & Robustness',
         'Status': 'COMPLIANT',
         'CREWS Implementation': 'AUC 0.7793, 5-fold CV stability (std 0.0037), calibration curve analysis, stress testing under 3 macroeconomic scenarios.'},
    ])
    st.table(eu_ai_data)

    st.markdown(
        "**Timeline:** EU AI Act entered into force August 2024. High-risk AI system "
        "requirements apply from August 2026 (24-month transition). CREWS documentation "
        "is designed to meet these requirements proactively."
    )

    st.markdown("[Back to top](#compliance-top)")
    st.markdown("---")

    # Section 8: FINMA Circular 2017/1
    st.subheader("8. FINMA Circular 2017/1 -- Model Risk Management (Switzerland)", anchor="finma-circular")

    st.markdown("""
**Applicability:** FINMA (Swiss Financial Market Supervisory Authority) Circular 2017/1
"Corporate Governance -- Banks" establishes model risk management requirements for
banks operating in Switzerland. While CREWS is a portfolio project, its design aligns
with FINMA expectations for credit risk models used in supervised institutions.

**Key FINMA Requirements and CREWS Alignment:**
""")

    finma_data = pd.DataFrame([
        {'FINMA Requirement': 'Model Inventory & Classification',
         'Principle': 'All models must be inventoried with risk tier classification',
         'CREWS Status': 'Model Governance section (Section 3) classifies CREWS as Tier 2 (Material). Model identification includes name, version, type, purpose, and date.'},
        {'FINMA Requirement': 'Independent Model Validation',
         'Principle': 'Models must be validated by a function independent from development',
         'CREWS Status': 'Pending -- listed in Production Readiness Checklist. 5-fold cross-validation and holdout test set provide quantitative validation. Second-line review recommended.'},
        {'FINMA Requirement': 'Model Documentation',
         'Principle': 'Complete documentation of methodology, assumptions, and limitations',
         'CREWS Status': 'Compliant -- model card, technical decisions log (13 decisions), 5 documented notebooks, SHAP analysis with business interpretation.'},
        {'FINMA Requirement': 'Ongoing Monitoring',
         'Principle': 'Regular performance monitoring and backtesting',
         'CREWS Status': 'Monitoring schedule defined (daily/weekly/monthly/quarterly/annual). PSI drift detection designed in AI agent. Automated revalidation pending.'},
        {'FINMA Requirement': 'Stress Testing',
         'Principle': 'Models must be stress tested under adverse scenarios',
         'CREWS Status': 'Compliant -- 3 stress scenarios implemented in Notebook 05 (interest rate shock, income reduction, employment stress). Results: +1.14% combined PD increase.'},
        {'FINMA Requirement': 'Board & Senior Management Reporting',
         'Principle': 'Regular model risk reporting to governance bodies',
         'CREWS Status': 'Executive summary and AI agent reports designed for senior management consumption. Dashboard provides real-time portfolio view.'},
    ])
    st.table(finma_data)

    st.markdown(
        "**Note:** FINMA Circular 2017/1 is supplemented by FINMA Guidance 03/2024 on "
        "artificial intelligence, which extends model risk management requirements to "
        "AI/ML models specifically. CREWS's dual explainability approach (SHAP + LIME) "
        "aligns with FINMA's emphasis on model interpretability for AI systems."
    )

    st.markdown("[Back to top](#compliance-top)")
    st.markdown("---")

    # Section 9: Swiss nDSG (Data Protection)
    st.subheader("9. Swiss nDSG -- Federal Act on Data Protection", anchor="swiss-ndsg")

    st.markdown("""
**Effective Date:** September 1, 2023 (revised Swiss Federal Act on Data Protection, nDSG/nFADP)

The nDSG modernizes Swiss data protection to align with GDPR standards while maintaining
Swiss-specific provisions. Credit scoring models processing personal data of Swiss residents
must comply with the following requirements:
""")

    ndsg_data = pd.DataFrame([
        {'nDSG Provision': 'Art. 21 -- Automated Individual Decisions',
         'Requirement': 'Data subjects have the right to be informed when a decision is based solely on automated processing that significantly affects them, and to request human review',
         'CREWS Alignment': 'SHAP waterfall plots provide per-decision explanations. Business threshold allows human override. Right-to-Explanation capability (Section 5) generates adverse action notices.'},
        {'nDSG Provision': 'Art. 25-27 -- Right to Information',
         'Requirement': 'Data subjects can request information about what personal data is processed, for what purpose, and the logic behind automated decisions',
         'CREWS Alignment': 'Model card documents all input features, purpose, and methodology. SHAP + LIME provide dual-method logic explanations. Feature names mapped to plain-English labels.'},
        {'nDSG Provision': 'Art. 22 -- Data Protection Impact Assessment (DPIA)',
         'Requirement': 'Required when processing poses a high risk to personality or fundamental rights. Credit scoring is explicitly high-risk.',
         'CREWS Alignment': 'Fair lending analysis (Section 2) identifies protected attributes. CODE_GENDER flagged as HIGH risk. DPIA documentation recommended before production deployment.'},
        {'nDSG Provision': 'Art. 6 -- Profiling',
         'Requirement': 'High-risk profiling (automated assessment of creditworthiness) requires explicit consent or legal basis. Must be transparent and proportionate.',
         'CREWS Alignment': 'CREWS performs high-risk profiling (credit scoring). Requires explicit consent or contractual necessity as legal basis. Feature selection documented with proportionality rationale.'},
        {'nDSG Provision': 'Art. 8 -- Data Security',
         'Requirement': 'Appropriate technical and organizational measures to protect personal data',
         'CREWS Alignment': 'Model artifacts stored locally. API keys in .env (not committed). .gitignore excludes sensitive files. Production deployment would require encryption at rest and in transit.'},
    ])
    st.table(ndsg_data)

    st.markdown("""
**Key Distinction from GDPR:**
- The nDSG applies to natural persons only (GDPR also covers some legal persons)
- The nDSG uses a **risk-based approach** similar to GDPR but with Swiss enforcement (FDPIC)
- Cross-border data transfer requires adequate protection level (Switzerland recognized by EU, but reverse adequacy requires assessment per Art. 16 nDSG)
- Penalties: up to CHF 250,000 for individuals (not organizations, unlike GDPR's percentage-of-revenue fines)

**Recommendation:** Before deploying CREWS in a Swiss banking context, conduct a formal DPIA
per Art. 22 nDSG and obtain legal review of the processing legal basis under Art. 6.
""")

    st.markdown("[Back to top](#compliance-top)")

    render_footer()
