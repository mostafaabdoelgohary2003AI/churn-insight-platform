# RUN-RESULTS (real runs, Nov 26 2026)

## python -m src.train (Iranian Churn Dataset, UCI #563, 3,150 real customers)
- rows: 3150, features: 15, churn_rate: 0.1571, held-out: 788 rows (25% stratified)
- hist gradient boosting: ROC-AUC 0.9924, PR-AUC 0.9636, accuracy 0.9632 (selected)
- random forest: ROC-AUC 0.9856, PR-AUC 0.8997, accuracy 0.9518
- logistic regression: ROC-AUC 0.9193, PR-AUC 0.7382, accuracy 0.8934
- held-out confusion (selected): TN 649, FP 15, FN 14, TP 110

## pytest -q
5 passed (cleaning, feature engineering, RF sanity AUC > 0.9, metrics thresholds, API predict+history)

## live API check (uvicorn, SQLite mode)
- GET /health -> {"status": "ok"}
- POST /predict {2 months, complaints, 25 call failures} -> churn_probability high (logged)
- POST /predict {48 months, heavy usage, no complaints} -> churn_probability low (logged)
- GET /history -> predictions logged with timestamps

## dashboard
results/dashboard.png - real screenshot of the Streamlit dashboard
(KPI cards, churn drivers, feature importances, scoring form).
