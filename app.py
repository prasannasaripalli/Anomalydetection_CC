import os
import requests
import streamlit as st
from src.utils import FEATURES

FUNC_URL = os.getenv("FUNC_URL", "").rstrip("/")
FUNC_KEY = os.getenv("FUNC_KEY", "")

st.set_page_config(page_title="CC Fraud Detector", layout="wide")
st.title("CC Fraud Detector")

cols = st.columns(3)
inputs = {}
for i, f in enumerate(FEATURES):
    with cols[i % 3]:
        step = 1.0 if f in ("Time", "Amount") else 0.01
        inputs[f] = st.number_input(f, value=0.0, step=step, format="%.2f")

if st.button("Predict", type="primary"):
    if not FUNC_URL:
        st.error("Set FUNC_URL environment variable")
        st.stop()

    url = FUNC_URL if not FUNC_KEY else f"{FUNC_URL}?code={FUNC_KEY}"
    r = requests.post(url, json=inputs, timeout=30)

    if r.status_code != 200:
        st.error(r.text)
        st.stop()

    out = r.json()

    if out.get("is_fraud"):
        st.error("Anomaly detected")
    else:
        st.success("Looks normal")

    st.metric("Anomaly score", f"{out.get('anomaly_score', 0):.6f}")
    st.write("Email sent:", out.get("email_sent", False))