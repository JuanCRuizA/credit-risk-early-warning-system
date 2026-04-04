# CREWS — Claude Code Persistent Context

## Project Overview
**CREWS** (Credit Risk Early Warning System) — XGBoost-based credit risk modeling on the Home Credit dataset. Deployed on Streamlit Cloud. Part of a banking data science portfolio targeting roles at international banks (UK, US, Switzerland).

## Project Path
`C:\Users\carlo\Documents\4.DS\credit-risk-early-warning-system`

## Tech Stack
XGBoost, LightGBM, SHAP, pandas, Streamlit, Python

## Project Structure
Four notebooks completed or in progress:
| Notebook | Topic | Status |
|---|---|---|
| NB01 | EDA | Complete |
| NB02 | Feature Engineering | Complete — 90+ features across three aggregation layers; includes `FEATURE_NARRATIVES` dictionary for future AI agent |
| NB03 | XGBoost Modeling | Complete — includes Brier Score, decile calibration table, PSI baseline monitoring, and Model Card JSON |
| NB04 | SHAP Explainability | In progress |
| NB05 | (pending) | — |

**Roadmap (pending):** NB05, Streamlit dashboard, AI agent notebook as capstone.

## Working Principles
- Ask before modifying existing cells; new sections can be added directly
- Do not restructure existing code without confirmation
- Keep outputs agent-ready: structured exports, named dictionaries
- Every technical decision should connect to banking regulatory logic when relevant:
  - Basel III, SR 11-7, IFRS 9, GDPR Art. 22, EU AI Act
  - FINMA 2017/1 and FINMA 2023/1, Swiss nDSG

## Language
All code, comments, and portfolio deliverables must be in **English**. Conversation may occasionally include small fragments in Spanish, but all outputs remain in English.

## Session Hygiene
At the end of every session, when the user says "close", "wrap up", or pastes a closing instruction, save a session summary to `docs/chats/YYYY-MM-DD_topic.md` with the following sections:
- Session title
- Date
- Status
- Deliverables
- Key decisions
- Next steps
- Context for next session

Create `docs/chats/` if it doesn't exist and add it to `.gitignore`.
