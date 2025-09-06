#!/usr/bin/env python3
# app_fastapi.py
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Spam Detector API", version="1.0.0", description="Real-time spam detection with scikit-learn pipeline.")

class PredictRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.5

class PredictResponse(BaseModel):
    label: str
    probability: float
    threshold: float

_pipeline = None

def get_pipeline(path="spam_detection_model.joblib"):
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(path)
    return _pipeline

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pipe = get_pipeline()
    proba = float(pipe.predict_proba([req.text])[0, 1])
    label = "Spam" if proba >= (req.threshold or 0.5) else "Ham"
    return PredictResponse(label=label, probability=proba, threshold=req.threshold or 0.5)
