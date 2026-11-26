"""Streamlit analytics dashboard for churn - logo-palette professional layout."""
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from prep import load_raw, engineer

DARK, PRIMARY, ACCENT, ALERT, LIGHT = "#006070", "#00808c", "#00c0c0", "#d45d4a", "#e8fafa"
SEQ = [[0, ACCENT], [1, DARK]]

st.set_page_config(page_title="Churn Insight Platform", layout="wide", initial_sidebar_state="collapsed")
st.markdown(f"""
<style>
  #MainMenu, header, footer {{visibility: hidden;}}
  .block-container {{padding-top: 1.6rem; max-width: 1240px;}}
  h1 {{font-size: 1.85rem !important; font-weight: 700 !important; color: {DARK} !important; margin-bottom: 0;}}
  .subtitle {{color: #5b6b73; font-size: 0.92rem; margin-bottom: 1.2rem;}}
  h2 {{font-size: 1.12rem !important; font-weight: 650 !important; color: {DARK} !important;
      margin-top: 1.5rem; margin-bottom: 0.4rem; border-left: 4px solid {ACCENT}; padding-left: 10px;}}
  .kpi {{background: #ffffff; border: 1px solid #d9ecec; border-radius: 12px; padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,96,112,0.07); border-top: 3px solid {ACCENT};}}
  .kpi .label {{color: #5b6b73; font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.05em;}}
  .kpi .value {{color: {DARK}; font-size: 1.65rem; font-weight: 700; margin-top: 2px;}}
  div[data-testid="stForm"] {{background: #ffffff; border: 1px solid #d9ecec; border-radius: 12px;
        padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,96,112,0.07);}}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Churn Insight Platform</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real telecom customer churn - 3,150 customers, Iranian Churn Dataset (UCI ML Repository #563)</div>',
            unsafe_allow_html=True)

df = engineer(load_raw())
with open("results/metrics.json") as f:
    m = json.load(f)
try:
    cv = json.load(open("results/benchmark_cv.json"))["iranian_churn"]["cv"]
except Exception:
    cv = None
sel = m["models"][m["selected_model"]]
LAY = dict(template="plotly_white", height=300)
MARG = dict(l=10, r=10, t=42, b=10)

c1, c2, c3, c4 = st.columns(4, gap="medium")
for col, label, value in [
    (c1, "Customers", f"{m['rows']:,}"),
    (c2, "Churn rate", f"{m['churn_rate']:.1%}"),
    (c3, "ROC-AUC (held-out)", f"{sel['roc_auc']:.4f}"),
    (c4, "Accuracy (held-out)", f"{sel['accuracy']:.1%}"),
]:
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div></div>',
                 unsafe_allow_html=True)

st.markdown("<h2>Churn drivers</h2>", unsafe_allow_html=True)
left, right = st.columns(2, gap="medium")
with left:
    g = df.groupby("complaints")["Churn"].mean().reset_index(name="churn_rate")
    g["complaints"] = g["complaints"].map({0: "No complaint", 1: "Complained"})
    fig = px.bar(g, x="complaints", y="churn_rate", color="churn_rate", color_continuous_scale=SEQ, **LAY)
    fig.update_layout(margin=MARG, title="Churn rate by complaint status", showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
with right:
    fig = px.histogram(df, x="seconds_of_use", color="Churn", nbins=40, barmode="overlay",
                       opacity=0.7, color_discrete_map={0: PRIMARY, 1: ALERT}, **LAY)
    fig.update_layout(margin=MARG, title="Usage (seconds of use) distribution by churn")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<h2>Segments and cohorts</h2>", unsafe_allow_html=True)
left, right = st.columns(2, gap="medium")
with left:
    df["subscription_cohort"] = pd.cut(df["subscription_length"], bins=[0, 6, 12, 24, 36, 48, 72],
                                       labels=["0-6m", "6-12m", "12-24m", "24-36m", "36-48m", "48m+"])
    g = df.groupby("subscription_cohort", observed=True)["Churn"].mean().reset_index(name="churn_rate")
    fig = px.line(g, x="subscription_cohort", y="churn_rate", markers=True, **LAY)
    fig.update_traces(line_color=DARK, marker_color=ACCENT, marker_size=9)
    fig.update_layout(margin=MARG, title="Churn rate by subscription cohort")
    st.plotly_chart(fig, use_container_width=True)
with right:
    g = df.groupby("age_group")["Churn"].mean().reset_index(name="churn_rate")
    g["age_group"] = g["age_group"].astype(str)
    fig = px.bar(g, x="age_group", y="churn_rate", color="churn_rate", color_continuous_scale=SEQ, **LAY)
    fig.update_layout(margin=MARG, title="Churn rate by age group", showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2, gap="medium")
with left:
    g = df.groupby("status")["Churn"].mean().reset_index(name="churn_rate")
    g["status"] = g["status"].map({1: "Active", 2: "Non-active"})
    fig = px.bar(g, x="status", y="churn_rate", color="churn_rate", color_continuous_scale=SEQ, **LAY)
    fig.update_layout(margin=MARG, title="Churn rate by account status", showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
with right:
    tf = pd.DataFrame(m["top_features"]).head(8).iloc[::-1]
    fig = px.bar(tf, x="importance", y="feature", orientation="h",
                 color_discrete_sequence=[DARK], **LAY)
    fig.update_layout(margin=MARG, title="Top feature importances (selected model)")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<h2>Model evaluation</h2>", unsafe_allow_html=True)
left, right = st.columns(2, gap="medium")
with left:
    if cv:
        rows = [{"model": k, "ROC-AUC": v["roc_auc_mean"], "Accuracy": v["accuracy_mean"]} for k, v in cv.items()]
        cdf = pd.DataFrame(rows).melt("model", var_name="metric", value_name="score")
        fig = px.bar(cdf, x="model", y="score", color="metric", barmode="group",
                     color_discrete_sequence=[DARK, ACCENT], **LAY)
        fig.update_layout(margin=MARG, title="Model comparison (5-fold cross-validation)")
        st.plotly_chart(fig, use_container_width=True)
with right:
    cm = sel["confusion_matrix"]
    labels = [["TN", "FP"], ["FN", "TP"]]
    fig = go.Figure(go.Heatmap(z=cm, x=["pred: stay", "pred: churn"], y=["actual: stay", "actual: churn"],
                               text=[[f"{labels[i][j]}<br>{cm[i][j]}" for j in range(2)] for i in range(2)],
                               texttemplate="%{text}", colorscale=SEQ, showscale=False))
    fig.update_layout(margin=MARG, title=f"Confusion matrix (held-out, {m['selected_model']})", **LAY)
    st.plotly_chart(fig, use_container_width=True)

if "roc_curves" in m:
    fig = go.Figure()
    colors = {"lr": "#8fa3ad", "hgb": DARK, "rf": ACCENT}
    for name, r in m["roc_curves"].items():
        fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], mode="lines", name=name,
                                 line=dict(color=colors.get(name, PRIMARY), width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                             line=dict(dash="dash", color="#999999")))
    fig.update_layout(margin=MARG, title="ROC curves (held-out)", xaxis_title="false positive rate",
                      yaxis_title="true positive rate", **LAY)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("<h2>Score a customer</h2>", unsafe_allow_html=True)
with st.form("score"):
    a, b, c = st.columns(3, gap="medium")
    sub = a.slider("Subscription length (months)", 1, 60, 12)
    sec = b.number_input("Seconds of use", 0, 20000, 500)
    comp = c.selectbox("Complaints filed", [0, 1])
    fails = a.slider("Call failures", 0, 40, 5)
    freq = b.number_input("Frequency of use", 0, 300, 20)
    value = c.number_input("Customer value", 0.0, 3000.0, 100.0)
    submitted = st.form_submit_button("Predict churn risk", use_container_width=True)
if submitted:
    import requests
    payload = {"subscription_length": sub, "seconds_of_use": sec, "complaints": comp,
               "call_failures": fails, "frequency_of_use": freq, "customer_value": value}
    try:
        r = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
        p = r.json()["churn_probability"]
        st.markdown(f'<div class="kpi"><div class="label">Predicted churn probability</div>'
                    f'<div class="value">{p:.1%}</div></div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"API not reachable at localhost:8000 ({e})")
