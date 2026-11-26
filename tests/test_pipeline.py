import json, os, sys
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prep import load_raw, engineer, load_xy
from src.train import build

def test_raw_loads_and_cleans():
    df = load_raw()
    assert len(df) == 3150
    assert df["Churn"].isin([0, 1]).all()
    assert "subscription_length" in df.columns

def test_feature_engineering():
    df = engineer(load_raw().head(50))
    for col in ["calls_per_month", "sms_ratio"]:
        assert col in df.columns

def test_model_trains_and_beats_baseline():
    X, y = load_xy()
    Xs = X.sample(1500, random_state=0); ys = y.loc[Xs.index]
    pipe = build("rf").fit(Xs, ys)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(ys, pipe.predict_proba(Xs)[:, 1]) > 0.9

def test_metrics_file_real():
    with open("results/metrics.json") as f:
        m = json.load(f)
    sel = m["models"][m["selected_model"]]
    assert sel["roc_auc"] > 0.95 and sel["accuracy"] > 0.93
    assert m["test_rows"] == 788

def test_api_predict_and_history(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_DB", str(tmp_path / "t.db"))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from fastapi.testclient import TestClient
    from src import api
    with TestClient(api.app) as client:
        r = client.post("/predict", json={"subscription_length": 2, "seconds_of_use": 50,
                                          "complaints": 1, "call_failures": 20})
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        r2 = client.get("/history")
        assert r2.status_code == 200 and len(r2.json()["predictions"]) >= 1
        assert client.get("/health").json()["status"] == "ok"
