# 🏦 Credit Risk Early Warning System with Autonomous AI Agent

An end-to-end credit risk assessment system that combines machine learning-based credit scoring with an autonomous AI agent for portfolio monitoring and early warning alerts. Built with XGBoost, SHAP, OpenAI API, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

## 🎯 Project Overview

This project demonstrates a complete Banking Data Science workflow, from data exploration to production deployment. The system:

1. **Predicts credit default probability** using gradient boosting models (XGBoost/LightGBM)
2. **Explains predictions** using SHAP values for regulatory compliance
3. **Monitors portfolio risk** through an autonomous AI agent
4. **Generates actionable alerts** when risk thresholds are breached
5. **Visualizes insights** through an interactive Streamlit dashboard

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│  Home Credit Dataset → Feature Engineering → Model Training │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ML MODEL LAYER                            │
│  XGBoost/LightGBM Credit Scoring + SHAP Explainability     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   AI AGENT LAYER                            │
│  Autonomous Risk Monitor (OpenAI API) → Alert Generation   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 PRESENTATION LAYER                          │
│  Streamlit Dashboard → Risk Reports → User Alerts          │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Dataset

This project uses the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset from Kaggle, containing:

- **307,511** loan applications
- **122** features including demographics, credit history, and payment behavior
- Real-world data from a consumer finance provider

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Explainability** | SHAP |
| **AI Agent** | OpenAI API, LangChain |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web App** | Streamlit |
| **Deployment** | Render.com |

## 📁 Project Structure

```
credit-risk-early-warning-system/
├── data/
│   ├── raw/                 # Original Kaggle data (not in repo)
│   └── processed/           # Cleaned, engineered features
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_explainability.ipynb
├── src/
│   ├── data_loader.py      # Data loading utilities
│   ├── features.py         # Feature engineering functions
│   ├── model.py            # Model training and prediction
│   └── explainability.py   # SHAP analysis functions
├── agents/
│   ├── risk_agent.py       # Autonomous risk monitoring agent
│   └── prompts/            # Agent prompt templates
├── app/
│   └── streamlit_app.py    # Dashboard application
├── reports/                 # Generated risk reports
├── config/                  # Configuration files
├── tests/                   # Unit tests
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Conda (recommended) or pip
- Kaggle account (for dataset download)
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/credit-risk-early-warning-system.git
cd credit-risk-early-warning-system

# Create conda environment
conda create -n credit-risk python=3.11 -y
conda activate credit-risk

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Download Dataset

1. Go to [Kaggle Home Credit Competition](https://www.kaggle.com/c/home-credit-default-risk/data)
2. Download `application_train.csv` and `application_test.csv`
3. Place files in `data/raw/`

### Run the Application

```bash
# Run Streamlit dashboard
streamlit run app/streamlit_app.py
```

## 📈 Results

*Results will be added upon project completion*

| Metric | Score |
|--------|-------|
| AUC-ROC | TBD |
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |

## 🤖 AI Agent Capabilities

The autonomous risk monitoring agent:

- ✅ Analyzes portfolio-level risk metrics
- ✅ Identifies high-risk customer segments
- ✅ Detects concentration risks
- ✅ Generates natural language risk reports
- ✅ Triggers alerts when thresholds are breached

## 📚 Credit Risk Fundamentals

This project implements the **Expected Loss** framework:

```
Expected Loss (EL) = PD × LGD × EAD
```

Where:
- **PD** (Probability of Default): Predicted by our ML model
- **LGD** (Loss Given Default): Percentage of exposure lost if default occurs
- **EAD** (Exposure at Default): Total amount at risk at default time

## 🙏 Acknowledgments

- [Home Credit](https://www.homecredit.net/) for providing the dataset
- Kaggle community for insights and kernels
- OpenAI for GPT API access

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Juan Carlos Ruiz Arteaga**

- LinkedIn: [Your LinkedIn]
- GitHub: [@your_username]

---

*This project was built as part of my journey to become a Banking Data Scientist, combining technical skills with a commitment to ethical AI and financial inclusion.*