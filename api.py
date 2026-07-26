# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

WORKDIR = Path(r"C:\Users\sujal\OneDrive\Documents\MINIP")  # adjust if needed

MODEL_PKL = WORKDIR / "final_heart_failure_xgb.pkl"
SCALER_PKL = WORKDIR / "scaler.pkl"
CSV = WORKDIR / "heart_failure_clinical_records_dataset.csv"

# Load model and scaler
model = joblib.load(MODEL_PKL)
scaler = joblib.load(SCALER_PKL)

# Infer feature order from CSV (exclude label column)
df = pd.read_csv(CSV)
label_names = {'death_event','death','DEATH_EVENT','target'}
features = [c for c in df.columns if c.lower() not in label_names]

# If your model expects a different order, replace `features` with exact list
print("Using feature order:", features)

# FastAPI app
app = FastAPI(title="HF Mortality Predictor")

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
        return (arr - mean) / (scale + 1e-12)
    # fallback: return raw
    return np.array(x_list, dtype=float)

def map_risk(prob):
    # same thresholds used earlier; tune if required
    if prob < 0.05: return "Very Low"
    if prob < 0.15: return "Low"
    if prob < 0.35: return "Moderate"
    if prob < 0.7: return "High"
    return "Very High"

@app.post("/predict")
def predict(inp: InputData):
    # expect inp.data to contain feature:value for all features
    try:
        x_list = [float(inp.data.get(f, 0.0)) for f in features]
        x_scaled = scale_input(x_list).reshape(1, -1)

        # model may be sklearn-like with predict_proba or produce single regression
        prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_scaled)
            # assume death class index is 1 (change if needed)
            prob = float(probs[0,1])
        else:
            # fallback: model.predict returns probability-like or numeric
            out = model.predict(x_scaled)
            prob = float(out[0])

        prob = max(0.0, min(1.0, prob))
        risk = map_risk(prob)

        return {"probability": prob, "risk": risk, "features_used": features}
    except Exception as e:
        return {"error": str(e)}
