#!/usr/bin/env python3
# app_streamlit.py
import joblib
import numpy as np
import streamlit as st
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Spam Detector", page_icon="📧")

@st.cache_resource
def load_pipeline(path="spam_detection_model.joblib"):
    return joblib.load(path)

st.title("📧 Real-time Spam Detection")
st.write("Paste a message or email below to classify it as **Spam** or **Ham**.")

threshold = st.slider("Spam probability threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

user_text = st.text_area("Message", height=200, placeholder="e.g., Congratulations! You've won a prize. Click here to claim...")

if st.button("Predict"):
    try:
        pipe = load_pipeline()
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        st.stop()

    if not user_text.strip():
        st.warning("Please enter some text.")
        st.stop()

    proba = float(pipe.predict_proba([user_text])[0, 1])
    pred = int(proba >= threshold)
    label = "Spam" if pred == 1 else "Ham"

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Spam probability: **{proba:.3f}**  (threshold={threshold:.2f})")
