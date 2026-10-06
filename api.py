# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

WORKDIR = Path(__file__).resolve().parent

MODEL_PKL = WORKDIR / "models" / "final_heart_failure_xgb.pkl"
SCALER_PKL = WORKDIR / "models" / "scaler.pkl"
CSV = WORKDIR / "data" / "heart_failure_clinical_records_dataset.csv"

# Load model and scaler
model = joblib.load(MODEL_PKL)
scaler = joblib.load(SCALER_PKL)

# Infer feature order (fallback to scaler or default list if CSV not present)
if CSV.exists():
    df = pd.read_csv(CSV)
    label_names = {'death_event', 'death', 'DEATH_EVENT', 'target'}
    features = [c for c in df.columns if c.lower() not in label_names]
elif hasattr(scaler, "feature_names_in_"):
    features = list(scaler.feature_names_in_)
else:
    features = [
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
        'ejection_fraction', 'high_blood_pressure', 'platelets',
        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
    ]

# FastAPI app
app = FastAPI(
    title="Heart Failure Mortality Prediction API",
    description="Machine Learning REST API for predicting heart failure mortality risk using clinical patient parameters.",
    version="1.0.0"
)

# allow local web page to call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    # accept a dict of feature:value pairs
    data: dict

def scale_input(x_list):
    # If scaler has mean_/scale_ apply StandardScaler behaviour
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = np.array(scaler.mean_)
        scale = np.array(scaler.scale_)
        arr = np.array(x_list, dtype=float)
        return ((arr - mean) / (scale + 1e-12)).reshape(1, -1)
    # fallback: return raw
    return np.array(x_list, dtype=float).reshape(1, -1)

def map_risk(prob):
    # 5-tier mortality risk stratification
    if prob < 0.05: return "Very Low"
    if prob < 0.15: return "Low"
    if prob < 0.35: return "Moderate"
    if prob < 0.70: return "High"
    return "Very High"

@app.get("/")
def root():
    return {
        "title": "Heart Failure Mortality Prediction API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "model": "XGBClassifier (Tuned)",
        "features": features
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "feature_count": len(features)
    }

@app.post("/predict")
def predict(inp: InputData):
    # expect inp.data to contain feature:value for all features
    try:
        if not inp.data:
            return {"error": "Input data dictionary cannot be empty."}

        x_list = [float(inp.data.get(f, 0.0)) for f in features]
        x_scaled = scale_input(x_list)

        # model may be sklearn-like with predict_proba or produce single regression
        prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_scaled)
            # assume death class index is 1
            prob = float(probs[0, 1])
        else:
            # fallback: model.predict returns probability-like or numeric
            out = model.predict(x_scaled)
            prob = float(out[0])

        prob = max(0.0, min(1.0, prob))
        risk = map_risk(prob)

        return {"probability": prob, "risk": risk, "features_used": features}
    except Exception as e:
        return {"error": str(e)}
