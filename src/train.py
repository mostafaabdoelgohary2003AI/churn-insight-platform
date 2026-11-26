"""Train, evaluate and persist the churn model. Writes results/metrics.json + plots."""
import json, os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix
from .prep import load_xy

def build(model):
    if model == "lr":
        return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))])
    if model == "hgb":
        return HistGradientBoostingClassifier(random_state=42)
    return RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced")

def main():
    os.makedirs("results", exist_ok=True)
    X, y = load_xy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    metrics = {"dataset": "Iranian Churn Dataset (UCI ML Repository #563), real telecom operator data",
               "rows": int(len(X)), "features": int(X.shape[1]),
               "churn_rate": round(float(y.mean()), 4), "test_rows": int(len(Xte)), "models": {}}
    best_name, best_auc, best_pipe = None, -1, None
    for name in ["lr", "hgb", "rf"]:
        pipe = build(name).fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = {"roc_auc": round(roc_auc_score(yte, proba), 4),
             "pr_auc": round(average_precision_score(yte, proba), 4),
             "accuracy": round(accuracy_score(yte, pred), 4),
             "confusion_matrix": confusion_matrix(yte, pred).tolist()}
        metrics["models"][name] = m
        if m["roc_auc"] > best_auc:
            best_name, best_auc, best_pipe = name, m["roc_auc"], pipe
    metrics["selected_model"] = best_name
    from sklearn.metrics import roc_curve
    roc = {}
    for name in ["lr", "hgb", "rf"]:
        p2 = build(name).fit(Xtr, ytr)
        proba2 = p2.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, proba2)
        step = max(1, len(fpr) // 60)
        roc[name] = {"fpr": [round(float(v), 4) for v in fpr[::step]],
                     "tpr": [round(float(v), 4) for v in tpr[::step]]}
    metrics["roc_curves"] = roc

    if hasattr(best_pipe, "feature_importances_"):
        imp = pd.Series(best_pipe.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    else:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(best_pipe, Xte, yte, n_repeats=5, random_state=42, scoring="roc_auc")
        imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False).head(10)
    metrics["top_features"] = [{"feature": k, "importance": round(float(v), 4)} for k, v in imp.items()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(list(imp.index)[::-1], list(imp.values)[::-1], color="#0B4B55")
    ax.set_title("Feature importance (selected model)")
    ax.set_xlabel("importance")
    fig.tight_layout(); fig.savefig("results/feature_importance.png", dpi=130)

    joblib.dump(best_pipe, "results/model.joblib")
    json.dump(metrics, open("results/metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
