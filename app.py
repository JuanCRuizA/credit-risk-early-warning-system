import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="CREWS - Credit Risk Early Warning System",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("Credit Risk Early Warning System (CREWS)")
st.markdown("**AI-Powered Portfolio Surveillance & Credit Risk Assessment**")
st.markdown("---")

# Sidebar - Individual Risk Calculator
st.sidebar.header("Individual Loan Risk Calculator")
st.sidebar.markdown("Enter borrower information:")

# Simplified inputs (3-4 key features)
ext_source_3 = st.sidebar.slider("External Credit Score (0-1)", 0.0, 1.0, 0.5, 0.01)
days_birth = st.sidebar.number_input("Age (years)", 20, 80, 35)
amt_credit = st.sidebar.number_input("Loan Amount ($)", 10000, 500000, 100000, 1000)
amt_annuity = st.sidebar.number_input("Monthly Payment ($)", 500, 10000, 2500, 100)

# Simple risk calculation (placeholder)
if st.sidebar.button("Calculate Risk", type="primary"):
    # Mock prediction (replace with actual model later if time permits)
    risk_score = 0.3 + (1 - ext_source_3) * 0.4 + np.random.uniform(-0.1, 0.1)
    risk_score = max(0, min(1, risk_score))
    
    st.sidebar.markdown("### Risk Assessment")
    
    # Color-coded risk display
    if risk_score < 0.3:
        st.sidebar.success(f"**Low Risk:** {risk_score:.1%}")
    elif risk_score < 0.59:
        st.sidebar.warning(f"**Moderate Risk:** {risk_score:.1%}")
    else:
        st.sidebar.error(f"**High Risk:** {risk_score:.1%}")
    
    st.sidebar.info(f"Expected Credit Loss: ${amt_credit * risk_score * 0.45:,.0f}")

# Main Dashboard - Tabs
tab1, tab2, tab3 = st.tabs(["Portfolio Overview", "AI Agent Insights", "Model Performance"])

# TAB 1: Portfolio Overview
with tab1:
    st.header("Portfolio Health Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Loans",
            value="307,511"
        )
    
    with col2:
        st.metric(
            label="High Risk Loans",
            value="42,073",
            delta="-13.68% of portfolio"
        )
    
    with col3:
        st.metric(
            label="Avg Default Probability",
            value="8.07%",
            delta="baseline rate"
        )
    
    with col4:
        st.metric(
            label="Model Status",
            value="Recalibration Needed"
        )
    
    st.markdown("---")
    
    # Risk distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Distribution")
        
        # REAL DATA from notebook 05 - Risk categories based on PD thresholds
        risk_data = pd.DataFrame({
            'Risk Category': ['Low Risk\n(PD < 0.3)', 'Moderate Risk\n(0.3 ≤ PD < 0.59)', 'High Risk\n(PD ≥ 0.59)'],
            'Count': [45833, 219605, 42073],  # Real counts from portfolio
            'Percentage': [14.9, 71.4, 13.68]  # Real percentages
        })
        
        fig = px.bar(
            risk_data,
            x='Risk Category',
            y='Count',
            text='Percentage',
            color='Risk Category',
            color_discrete_map={
                'Low Risk\n(PD < 0.3)': '#2ecc71',
                'Moderate Risk\n(0.3 ≤ PD < 0.59)': '#f39c12',
                'High Risk\n(PD ≥ 0.59)': '#e74c3c'
            }
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk by Loan Vintage")
        
        # Real vintage analysis from notebook
        vintage_data = pd.DataFrame({
            'Vintage': ['New\n(0-6 months)', 'Medium\n(6-18 months)', 'Mature\n(18+ months)'],
            'High Risk %': [42.7, 60.3, 63.8]
        })
        
        fig = px.line(
            vintage_data,
            x='Vintage',
            y='High Risk %',
            markers=True,
            line_shape='linear'
        )
        fig.update_traces(line_color='#e74c3c', marker=dict(size=12))
        fig.update_layout(height=400, yaxis_title="High Risk Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: AI Agent Insights
with tab2:
    st.header("Autonomous AI Agent Analysis")
    
    # Agent info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("**Last Analysis:** January 31, 2026 | **Status:** Completed | **Model:** Claude Sonnet 4")
    
    with col2:
        if st.button("View Full Report", type="secondary"):
            st.toast("Check notebooks/05_portfolio_surveillance.ipynb")
    
    st.markdown("---")
    
    # Display AI Agent screenshot
    st.image("docs/screenshots/ai_agent_overview.png", 
             caption="AI Agent Portfolio Surveillance Output",
             use_container_width=True)
    
    st.markdown("---")
    
    # Key findings in columns
    st.subheader("Key Findings & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
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
    
    with col2:
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
        """)
    
    # Regulatory alignment
    st.markdown("---")
    st.subheader("Regulatory Compliance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("""
        **SR 11-7 Alignment**
        - Ongoing monitoring
        - Model validation
        - Audit trail
        """)
    
    with col2:
        st.success("""
        **Basel III/IV**
        - Risk-weighted assets
        - Capital adequacy
        - Stress testing
        """)
    
    with col3:
        st.success("""
        **IFRS 9**
        - ECL provisioning
        - Forward-looking
        - Stage classification
        """)

# TAB 3: Model Performance
with tab3:
    st.header("Model Performance Metrics")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ROC-AUC", "0.76")
    
    with col2:
        st.metric("Gini Coefficient", "0.518")
    
    with col3:
        st.metric("KS Statistic", "0.421")
    
    with col4:
        st.metric("Brier Score", "0.156")
    
    st.markdown("---")
    
    # Model info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Architecture")
        st.markdown("""
        - **Algorithm:** XGBoost (Gradient Boosting)
        - **Features:** 200+ engineered features
        - **Training Data:** 307,511 loan applications
        - **Class Imbalance:** Handled via scale_pos_weight
        - **Explainability:** SHAP values for all predictions
        """)
        
        st.subheader("Feature Engineering")
        st.markdown("""
        - Aggregated features (sum, mean, max, min)
        - Ratio features (payment-to-income, credit-to-income)
        - Temporal features (loan age, days employed)
        - External data integration (credit bureau scores)
        - Interaction features between key variables
        """)
    
    with col2:
        st.subheader("Business Value")
        st.markdown("""
        - **Cost-sensitive evaluation** with profit optimization
        - **Regulatory compliance:** Basel III/IV, SR 11-7, IFRS 9
        - **Explainable AI:** SHAP analysis for audit trail
        - **Autonomous monitoring:** AI agent for portfolio surveillance
        - **Real-time risk assessment:** Streamlit dashboard deployment
        """)
        
        st.subheader("Technical Stack")
        st.markdown("""
        - Python 3.11+
        - XGBoost, scikit-learn, pandas, numpy
        - SHAP for explainability
        - Claude API for AI agent
        - Streamlit for deployment
        - GitHub for version control
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>CREWS - Credit Risk Early Warning System</strong></p>
    <p>Developed by Juan Carlos Ruiz Arteaga | Banking Data Scientist | MSc in Data Science & AI, University of Liverpool</p>
    <p>GitHub Repository | Contact: carlosarte11@gmail.com</p>
</div>
""", unsafe_allow_html=True)
