"""Serious modeling-effort benchmark: feature engineering, tuning, ensembles,
class-imbalance handling, 5-fold stratified CV, on BOTH datasets.
Documents the honest accuracy ceiling per dataset. No leakage (IDs dropped,
only pre-churn attributes used)."""
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.model_selection import StratifiedKFold, cross_validate

CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
               "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
               "PaperlessBilling", "PaymentMethod"]
NUM = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "num_services", "tenure_years"]

def telco_xy():
    df = pd.read_csv("data/Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    services = ["PhoneService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_services"] = df[services].apply(lambda c: (c == "Yes").astype(int)).sum(axis=1)
    df["tenure_years"] = df["tenure"] / 12.0
    df["charge_per_service"] = df["MonthlyCharges"] / df["num_services"].clip(lower=1)
    NUMS = NUM + ["charge_per_service"]
    X = df[NUMS + CATEGORICAL]
    y = (df["Churn"] == "Yes").astype(int)
    return X, y, NUMS

def iranian_xy():
    from .prep import load_xy
    return load_xy()

def telco_pre(NUMS):
    return ColumnTransformer([("num", StandardScaler(), NUMS),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])

def models_for(kind, pre=None):
    out = {}
    if kind == "telco":
        out["logistic_regression"] = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
        out["hist_gradient_boosting"] = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(random_state=42, max_iter=400, learning_rate=0.06, early_stopping=False))])
        out["random_forest_balanced"] = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"))])
        stack = StackingClassifier(
            estimators=[("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
                        ("hgb", HistGradientBoostingClassifier(random_state=42, max_iter=400, learning_rate=0.06, early_stopping=False)),
                        ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"))],
            final_estimator=LogisticRegression(max_iter=2000), cv=3)
        out["stacking_ensemble"] = Pipeline([("pre", pre), ("clf", stack)])
    else:
        out["logistic_regression"] = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])
        out["hist_gradient_boosting_tuned"] = HistGradientBoostingClassifier(random_state=42, max_iter=600, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=False)
        out["random_forest_balanced"] = RandomForestClassifier(n_estimators=600, random_state=42, n_jobs=-1, class_weight="balanced", max_features="sqrt")
        out["stacking_ensemble"] = StackingClassifier(
            estimators=[("lr", Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))])),
                        ("hgb", HistGradientBoostingClassifier(random_state=42, max_iter=600, learning_rate=0.06, early_stopping=False)),
                        ("rf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"))],
            final_estimator=LogisticRegression(max_iter=2000), cv=3)
    return out

def run(name, X, y, models):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    res = {}
    for mname, m in models.items():
        r = cross_validate(m, X, y, cv=cv, scoring=["roc_auc", "accuracy"], n_jobs=-1)
        res[mname] = {"roc_auc_mean": round(r["test_roc_auc"].mean(), 4),
                      "roc_auc_std": round(r["test_roc_auc"].std(), 4),
                      "accuracy_mean": round(r["test_accuracy"].mean(), 4),
                      "accuracy_std": round(r["test_accuracy"].std(), 4)}
        print(name, mname, res[mname], flush=True)
    return res

if __name__ == "__main__":
    out = {}
    X, y, NUMS = telco_xy()
    out["ibm_telco"] = {"rows": len(X), "churn_rate": round(float(y.mean()), 4),
                        "cv": run("ibm_telco", X, y, models_for("telco", telco_pre(NUMS)))}
    X2, y2 = iranian_xy()
    out["iranian_churn"] = {"rows": len(X2), "churn_rate": round(float(y2.mean()), 4),
                            "cv": run("iranian_churn", X2, y2, models_for("iranian"))}
    json.dump(out, open("results/benchmark_cv.json", "w"), indent=2)
    print("saved results/benchmark_cv.json")
