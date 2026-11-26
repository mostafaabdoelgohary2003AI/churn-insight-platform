<p align="center"><img src="assets/logo.png" width="128" alt="churn-insight-platform logo"></p>

# churn-insight-platform

End-to-end customer-churn data science platform on real telecom operator data: cleaning,
feature engineering, model comparison, explainability, a FastAPI inference service with a
Postgres prediction log, and a Streamlit analytics dashboard. Dockerized and deploy-ready.

Built with AI pair-programming (Claude). All numbers below are real outputs of the
commands shown, not targets.

## Dataset
Iranian Churn Dataset (UCI ML Repository #563) - the complete dataset of 3,150 real
customers from an Iranian telecom operator, 13 usage attributes + churn label (the models use 15 features: these plus 2 engineered ones)
(15.7% churn rate). Source: archive.ics.uci.edu/dataset/563. Included in `data/`.

## Real results (held-out 25%, 788 customers, stratified split, seed 42)
| model | ROC-AUC | PR-AUC | accuracy |
|---|---|---|---|
| histogram gradient boosting (selected) | 0.9924 | 0.9636 | 96.3% |
| random forest (400 trees, balanced) | 0.9856 | 0.8997 | 95.2% |
| logistic regression | 0.9193 | 0.7382 | 89.3% |

Leakage check: no IDs or post-churn fields are used. Two columns are close to the outcome,
though: `Status` (active / non-active) and `Complains`. Re-running the gradient-boosting
model on the same split without them still gives ROC-AUC 0.990 (0.990 without both,
accuracy 95.4%), so the score does not depend on them, but treat them as near-outcome
signals in a real deployment. The best of the three models was picked by its score on the
same held-out 25% (there is no separate validation split), so the 96.3% is slightly
optimistic.
Confusion matrices and top features: `results/metrics.json`, `results/feature_importance.png`.

## Run locally (no Docker)
    pip install -r requirements.txt
    python -m src.train            # trains, evaluates, writes results/
    pytest -q                      # 5 tests
    uvicorn src.api:app --port 8000
    streamlit run src/dashboard.py

## Run with Docker (Postgres included)
    docker compose up --build
    # api on :8000, dashboard on :8501, postgres 16 with migrations applied

Without Docker the API logs predictions to SQLite (`data/app.db`); with compose it uses
PostgreSQL (`DATABASE_URL`). Same schema, one migration file each.

## API
    POST /predict   one customer JSON -> churn probability + label (logged)
    POST /batch     up to 5,000 customers
    GET  /history   latest logged predictions
    GET  /metrics   training metrics
    GET  /health

## Dashboard
Streamlit app with KPI cards, churn-driver charts, segment and cohort views (subscription
cohorts, age groups, account status), model comparison (5-fold CV), held-out confusion
matrix and ROC curves, and a live scoring form. See `results/dashboard.png`.

## Honest limits
- 3,150 customers from one operator in one market; strong separability (high accuracy is
  documented for this dataset in the literature) but real deployments need company data
  and periodic retraining.
- 0.50 decision threshold is a default; tune to retention cost in production.
- Compose stack validated locally with SQLite (no Docker daemon in the build environment).

## Stack
Python, scikit-learn, pandas, FastAPI, Streamlit, Plotly, PostgreSQL (Docker) / SQLite,
pytest, GitHub Actions CI.

