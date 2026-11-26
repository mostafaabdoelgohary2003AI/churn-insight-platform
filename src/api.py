"""FastAPI inference service for the churn model."""
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from . import db
from .prep import engineer

app = FastAPI(title="churn-insight-platform", version="2.0.0")
MODEL_PATH = "results/model.joblib"
_pipe = None

def pipe():
    global _pipe
    if _pipe is None:
        _pipe = joblib.load(MODEL_PATH)
    return _pipe

class Customer(BaseModel):
    call_failures: int = 0
    complaints: int = 0
    subscription_length: int = 12
    charge_amount: int = 0
    seconds_of_use: int = 1000
    frequency_of_use: int = 20
    frequency_of_sms: int = 5
    distinct_called_numbers: int = 10
    age_group: int = 3
    tariff_plan: int = 1
    status: int = 1
    age: int = 30
    customer_value: float = 100.0

def _score(c: Customer) -> float:
    df = engineer(pd.DataFrame([c.dict()]))
    return float(pipe().predict_proba(df)[:, 1][0])

@app.on_event("startup")
def startup():
    db.run_migrations()

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH}

@app.post("/predict")
def predict(c: Customer):
    p = _score(c)
    pred = int(p >= 0.5)
    db.log_prediction(c.dict(), p, pred)
    return {"churn_probability": round(p, 4), "churn_prediction": pred}

@app.post("/batch")
def batch(customers: list[Customer]):
    if len(customers) > 5000:
        raise HTTPException(413, "max 5000 rows per batch")
    out = []
    for c in customers:
        p = _score(c)
        pred = int(p >= 0.5)
        db.log_prediction(c.dict(), p, pred)
        out.append({"churn_probability": round(p, 4), "churn_prediction": pred})
    return {"count": len(out), "predictions": out}

@app.get("/history")
def get_history(limit: int = 50):
    return {"predictions": db.history(limit)}

@app.get("/metrics")
def get_metrics():
    with open("results/metrics.json") as f:
        return json.load(f)
