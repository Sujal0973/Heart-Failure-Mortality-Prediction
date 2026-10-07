# api.py
"""
Heart Failure Mortality Prediction API
Production-grade FastAPI REST service for cardiovascular mortality risk stratification.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

WORKDIR = Path(__file__).resolve().parent

MODEL_PKL = WORKDIR / "models" / "final_heart_failure_xgb.pkl"
SCALER_PKL = WORKDIR / "models" / "scaler.pkl"
CSV = WORKDIR / "data" / "heart_failure_clinical_records_dataset.csv"
STATIC_DIR = WORKDIR / "static"

# Load model and scaler
if not MODEL_PKL.exists():
    raise FileNotFoundError(f"Model file not found at: {MODEL_PKL}")
if not SCALER_PKL.exists():
    raise FileNotFoundError(f"Scaler file not found at: {SCALER_PKL}")

model = joblib.load(MODEL_PKL)
scaler = joblib.load(SCALER_PKL)

# Feature list
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

# Feature importance map from tuned XGBoost
FEATURE_IMPORTANCES = {
    'time': 0.2805,
    'serum_creatinine': 0.1431,
    'ejection_fraction': 0.1294,
    'age': 0.0636,
    'platelets': 0.0565,
    'creatinine_phosphokinase': 0.0541,
    'sex': 0.0538,
    'anaemia': 0.0474,
    'diabetes': 0.0463,
    'smoking': 0.0452,
    'serum_sodium': 0.0426,
    'high_blood_pressure': 0.0374
}

CLINICAL_METADATA = {
    "age": {
        "label": "Age",
        "unit": "years",
        "type": "continuous",
        "normal_range": "40 – 95",
        "clinical_significance": "Independent risk factor; elderly patients have reduced physiological cardiac reserve.",
        "min": 40.0, "max": 95.0, "default": 60.0
    },
    "anaemia": {
        "label": "Anaemia",
        "unit": "binary (0=No, 1=Yes)",
        "type": "categorical",
        "normal_range": "Absent (0)",
        "clinical_significance": "Reduces tissue oxygenation and increases hemodynamic myocardial workload.",
        "min": 0, "max": 1, "default": 0
    },
    "creatinine_phosphokinase": {
        "label": "Creatinine Phosphokinase (CPK)",
        "unit": "mcg/L",
        "type": "continuous",
        "normal_range": "10 – 120 mcg/L",
        "clinical_significance": "Intracellular muscle/cardiac enzyme released during cellular and myocardial injury.",
        "min": 20.0, "max": 8000.0, "default": 200.0
    },
    "diabetes": {
        "label": "Diabetes Mellitus",
        "unit": "binary (0=No, 1=Yes)",
        "type": "categorical",
        "normal_range": "Absent (0)",
        "clinical_significance": "Diabetic micro- and macro-vascular complications accelerate heart failure progression.",
        "min": 0, "max": 1, "default": 0
    },
    "ejection_fraction": {
        "label": "Ejection Fraction",
        "unit": "%",
        "type": "continuous",
        "normal_range": "50% – 70%",
        "clinical_significance": "Percentage of ventricular blood volume pumped per stroke; <40% defines HFrEF.",
        "min": 14.0, "max": 80.0, "default": 50.0
    },
    "high_blood_pressure": {
        "label": "Hypertension",
        "unit": "binary (0=No, 1=Yes)",
        "type": "categorical",
        "normal_range": "Absent (0)",
        "clinical_significance": "Sustained arterial hypertension causes left ventricular hypertrophy and cardiac remodeling.",
        "min": 0, "max": 1, "default": 0
    },
    "platelets": {
        "label": "Platelet Count",
        "unit": "kiloplatelets/mL",
        "type": "continuous",
        "normal_range": "150,000 – 450,000 /mL",
        "clinical_significance": "Thrombocyte integrity indicates coagulation balance and systemic inflammation.",
        "min": 25000.0, "max": 850000.0, "default": 263000.0
    },
    "serum_creatinine": {
        "label": "Serum Creatinine",
        "unit": "mg/dL",
        "type": "continuous",
        "normal_range": "0.7 – 1.3 mg/dL",
        "clinical_significance": "Renal filtration marker; elevated levels indicate cardiorenal syndrome and poor prognosis.",
        "min": 0.5, "max": 10.0, "default": 1.0
    },
    "serum_sodium": {
        "label": "Serum Sodium",
        "unit": "mEq/L",
        "type": "continuous",
        "normal_range": "135 – 145 mEq/L",
        "clinical_significance": "Hyponatremia (<135 mEq/L) reflects severe neurohormonal activation and hemodynamic congestion.",
        "min": 110.0, "max": 150.0, "default": 137.0
    },
    "sex": {
        "label": "Sex",
        "unit": "binary (0=Female, 1=Male)",
        "type": "categorical",
        "normal_range": "N/A",
        "clinical_significance": "Cardiovascular epidemiology and risk trajectories vary across biological sex.",
        "min": 0, "max": 1, "default": 1
    },
    "smoking": {
        "label": "Smoking Habit",
        "unit": "binary (0=No, 1=Yes)",
        "type": "categorical",
        "normal_range": "Non-smoker (0)",
        "clinical_significance": "Nicotine and combustion toxins exacerbate endothelial dysfunction and oxidative stress.",
        "min": 0, "max": 1, "default": 0
    },
    "time": {
        "label": "Follow-up Window",
        "unit": "days",
        "type": "continuous",
        "normal_range": "4 – 285 days",
        "clinical_significance": "Observation window; patients with shorter time face higher vulnerability to early acute events.",
        "min": 4.0, "max": 285.0, "default": 100.0
    }
}

RISK_TIERS = {
    "Very Low": {
        "range": "< 5%",
        "color": "#10b981",
        "badge_bg": "rgba(16, 185, 129, 0.15)",
        "badge_border": "rgba(16, 185, 129, 0.35)",
        "action": "Maintain optimal medication adherence, healthy diet, and routine annual follow-up."
    },
    "Low": {
        "range": "5% – 15%",
        "color": "#38bdf8",
        "badge_bg": "rgba(56, 189, 248, 0.15)",
        "badge_border": "rgba(56, 189, 248, 0.35)",
        "action": "Implement dietary sodium restriction (<2g/day), track daily weight, and attend routine outpatient clinic visits."
    },
    "Moderate": {
        "range": "15% – 35%",
        "color": "#f59e0b",
        "badge_bg": "rgba(245, 158, 11, 0.15)",
        "badge_border": "rgba(245, 158, 11, 0.35)",
        "action": "Schedule clinical medication review (beta-blocker, ACEi/ARNI), repeat renal panel, and keep a daily symptom diary."
    },
    "High": {
        "range": "35% – 70%",
        "color": "#f97316",
        "badge_bg": "rgba(249, 115, 22, 0.15)",
        "badge_border": "rgba(249, 115, 22, 0.35)",
        "action": "Prompt cardiology review recommended within 48-72h. Monitor for rapid fluid accumulation (>2kg in 48h) or orthopnea."
    },
    "Very High": {
        "range": "≥ 70%",
        "color": "#ef4444",
        "badge_bg": "rgba(239, 68, 68, 0.15)",
        "badge_border": "rgba(239, 68, 68, 0.35)",
        "action": "Urgent specialist evaluation required. High risk of acute decompensation; assess for immediate hospitalization or ICU care."
    }
}

# Pydantic Models
class PatientRecord(BaseModel):
    age: float = Field(default=60.0, ge=18.0, le=120.0, description="Age in years", examples=[65.0])
    anaemia: int = Field(default=0, ge=0, le=1, description="Anaemia indicator (0=No, 1=Yes)", examples=[0])
    creatinine_phosphokinase: float = Field(default=200.0, ge=5.0, le=20000.0, description="CPK enzyme (mcg/L)", examples=[160.0])
    diabetes: int = Field(default=0, ge=0, le=1, description="Diabetes mellitus (0=No, 1=Yes)", examples=[0])
    ejection_fraction: float = Field(default=50.0, ge=5.0, le=90.0, description="Ejection fraction percentage (%)", examples=[35.0])
    high_blood_pressure: int = Field(default=0, ge=0, le=1, description="Hypertension history (0=No, 1=Yes)", examples=[1])
    platelets: float = Field(default=263000.0, ge=1000.0, le=1500000.0, description="Platelets count (per mL)", examples=[263000.0])
    serum_creatinine: float = Field(default=1.0, ge=0.1, le=20.0, description="Serum creatinine (mg/dL)", examples=[1.3])
    serum_sodium: float = Field(default=137.0, ge=90.0, le=170.0, description="Serum sodium (mEq/L)", examples=[136.0])
    sex: int = Field(default=1, ge=0, le=1, description="Biological sex (0=Female, 1=Male)", examples=[1])
    smoking: int = Field(default=0, ge=0, le=1, description="Smoking history (0=No, 1=Yes)", examples=[0])
    time: float = Field(default=100.0, ge=1.0, le=500.0, description="Follow-up window in days", examples=[90.0])

class InputData(BaseModel):
    # Backward compatible: accept {"data": {...}} or direct patient dictionary
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Patient feature dictionary mapping clinical parameter name to value",
        examples=[{
            "age": 65,
            "anaemia": 0,
            "creatinine_phosphokinase": 160,
            "diabetes": 0,
            "ejection_fraction": 35,
            "high_blood_pressure": 1,
            "platelets": 263000,
            "serum_creatinine": 1.3,
            "serum_sodium": 136,
            "sex": 1,
            "smoking": 0,
            "time": 90
        }]
    )

class BatchInputData(BaseModel):
    patients: List[Dict[str, Any]] = Field(
        ...,
        description="List of patient dictionaries to evaluate in batch",
        examples=[[
            {"age": 45, "anaemia": 0, "creatinine_phosphokinase": 200, "diabetes": 0, "ejection_fraction": 60, "high_blood_pressure": 0, "platelets": 300000, "serum_creatinine": 0.9, "serum_sodium": 140, "sex": 1, "smoking": 0, "time": 250},
            {"age": 75, "anaemia": 1, "creatinine_phosphokinase": 582, "diabetes": 1, "ejection_fraction": 25, "high_blood_pressure": 1, "platelets": 200000, "serum_creatinine": 2.5, "serum_sodium": 131, "sex": 1, "smoking": 1, "time": 30}
        ]]
    )

# App Setup
app = FastAPI(
    title="Heart Failure Mortality Prediction API",
    description=(
        "Production-grade Clinical Machine Learning REST API for Heart Failure Mortality "
        "Risk Stratification using XGBoost and standard clinical laboratory markers."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static frontend files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def scale_input(x_list: List[float]) -> np.ndarray:
    """Scales patient feature vector using the fitted StandardScaler."""
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        mean = np.array(scaler.mean_)
        scale = np.array(scaler.scale_)
        arr = np.array(x_list, dtype=float)
        return ((arr - mean) / (scale + 1e-12)).reshape(1, -1)
    return np.array(x_list, dtype=float).reshape(1, -1)

def map_risk(prob: float) -> str:
    """5-tier calibrated mortality risk stratification."""
    if prob < 0.05:
        return "Very Low"
    if prob < 0.15:
        return "Low"
    if prob < 0.35:
        return "Moderate"
    if prob < 0.70:
        return "High"
    return "Very High"

def identify_clinical_drivers(patient_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Analyzes patient metrics against clinical guidelines to pinpoint primary risk contributors."""
    drivers = []
    
    ef = float(patient_data.get('ejection_fraction', 50))
    if ef < 30:
        drivers.append({
            "feature": "ejection_fraction",
            "severity": "critical",
            "title": "Severe Systolic Heart Failure (EF < 30%)",
            "detail": f"Current EF is {ef:.0f}%. Severely reduced ventricular contractility substantially amplifies mortality."
        })
    elif ef < 40:
        drivers.append({
            "feature": "ejection_fraction",
            "severity": "high",
            "title": "Reduced Ejection Fraction (HFrEF)",
            "detail": f"Current EF is {ef:.0f}%. Moderate ventricular impairment (normal: 50–70%)."
        })

    sc = float(patient_data.get('serum_creatinine', 1.0))
    if sc >= 2.0:
        drivers.append({
            "feature": "serum_creatinine",
            "severity": "critical",
            "title": "Severe Renal Dysfunction (Creatinine ≥ 2.0 mg/dL)",
            "detail": f"Current creatinine is {sc:.2f} mg/dL (normal: 0.7–1.3). High cardiorenal syndrome vulnerability."
        })
    elif sc > 1.3:
        drivers.append({
            "feature": "serum_creatinine",
            "severity": "high",
            "title": "Elevated Serum Creatinine",
            "detail": f"Current creatinine is {sc:.2f} mg/dL indicates compromised glomerular filtration."
        })

    ss = float(patient_data.get('serum_sodium', 137))
    if ss < 135:
        drivers.append({
            "feature": "serum_sodium",
            "severity": "high",
            "title": "Hyponatremia Alert (< 135 mEq/L)",
            "detail": f"Serum sodium is {ss:.1f} mEq/L (normal: 135–145). Strongly linked to neurohormonal hyperactivation."
        })

    age = float(patient_data.get('age', 60))
    if age >= 75:
        drivers.append({
            "feature": "age",
            "severity": "moderate",
            "title": "Geriatric Age Risk",
            "detail": f"Patient age is {age:.0f} years; diminished physiologic reserve increases adverse cardiovascular outcomes."
        })

    t = float(patient_data.get('time', 100))
    if t < 30:
        drivers.append({
            "feature": "time",
            "severity": "high",
            "title": "Acute Post-Onset Window (< 30 days)",
            "detail": f"Follow-up is {t:.0f} days. Early monitoring phases carry the highest frequency of decompensation."
        })

    if int(patient_data.get('high_blood_pressure', 0)) == 1:
        drivers.append({
            "feature": "high_blood_pressure",
            "severity": "moderate",
            "title": "Hypertensive Comorbidity",
            "detail": "Chronic afterload elevation promotes ventricular remodeling and hypertrophy."
        })

    if int(patient_data.get('diabetes', 0)) == 1:
        drivers.append({
            "feature": "diabetes",
            "severity": "moderate",
            "title": "Diabetic Cardiopathy Risk",
            "detail": "Accelerates microvascular damage and autonomic cardiac dysfunction."
        })

    if int(patient_data.get('anaemia', 0)) == 1:
        drivers.append({
            "feature": "anaemia",
            "severity": "moderate",
            "title": "Anaemic Hemodynamic Strain",
            "detail": "Compromised hemoglobin requires higher cardiac output to maintain peripheral oxygenation."
        })

    if int(patient_data.get('smoking', 0)) == 1:
        drivers.append({
            "feature": "smoking",
            "severity": "moderate",
            "title": "Active Smoking Habit",
            "detail": "Promotes endothelial dysfunction, plaque instability, and vasoconstriction."
        })

    cpk = float(patient_data.get('creatinine_phosphokinase', 200))
    if cpk > 500:
        drivers.append({
            "feature": "creatinine_phosphokinase",
            "severity": "moderate",
            "title": "Elevated CPK Enzyme Level",
            "detail": f"CPK level is {cpk:.0f} mcg/L (normal: 10–120), indicating cellular injury or skeletal/myocardial strain."
        })

    return drivers

def compute_patient_prediction(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Internal helper to score a single patient dictionary."""
    # Convert and extract features
    x_list = [float(patient_dict.get(f, CLINICAL_METADATA.get(f, {}).get('default', 0.0))) for f in features]
    x_scaled = scale_input(x_list)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_scaled)
        prob = float(probs[0, 1])
    else:
        out = model.predict(x_scaled)
        prob = float(out[0])

    prob = max(0.0, min(1.0, prob))
    risk = map_risk(prob)
    tier_info = RISK_TIERS[risk]
    drivers = identify_clinical_drivers(patient_dict)

    return {
        "probability": round(prob, 4),
        "probability_percentage": f"{round(prob * 100, 1)}%",
        "risk": risk,
        "risk_color": tier_info["color"],
        "action_protocol": tier_info["action"],
        "key_drivers": drivers,
        "features_used": features
    }

# Routes
@app.get("/")
def root():
    return {
        "title": "Heart Failure Mortality Prediction API",
        "version": "2.0.0",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "dashboard_url": "/dashboard",
        "model": "XGBClassifier (Optimized & Calibrated)",
        "features": features
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "feature_count": len(features),
        "version": "2.0.0"
    }

@app.get("/features")
def get_features():
    """Returns detailed clinical parameter metadata, normal ranges, and XGBoost importance ratings."""
    feature_list = []
    for f in features:
        meta = CLINICAL_METADATA.get(f, {})
        feature_list.append({
            "name": f,
            "label": meta.get("label", f),
            "unit": meta.get("unit", ""),
            "type": meta.get("type", "continuous"),
            "normal_range": meta.get("normal_range", "N/A"),
            "clinical_significance": meta.get("clinical_significance", ""),
            "min": meta.get("min", 0),
            "max": meta.get("max", 100),
            "default": meta.get("default", 0),
            "importance_weight": FEATURE_IMPORTANCES.get(f, 0.0)
        })
    return {"features": feature_list}

@app.get("/model/info")
def get_model_info():
    """Returns model architecture parameters, performance metrics, and feature importance rankings."""
    importance_rankings = [
        {"feature": k, "importance_percentage": round(v * 100, 2), "label": CLINICAL_METADATA.get(k, {}).get("label", k)}
        for k, v in sorted(FEATURE_IMPORTANCES.items(), key=lambda x: x[1], reverse=True)
    ]
    return {
        "algorithm": "Extreme Gradient Boosting (XGBoost)",
        "objective": "binary:logistic",
        "hyperparameters": {
            "n_estimators": 200,
            "learning_rate": 0.01,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 3.0,
            "eval_metric": "logloss",
            "random_state": 42
        },
        "evaluation_metrics": {
            "holdout_test_accuracy": "83.3%",
            "holdout_roc_auc": 0.879,
            "mortality_recall_sensitivity": "68.4%",
            "survivor_recall_specificity": "90.2%",
            "precision_mortality": 0.76,
            "f1_score_mortality": 0.72
        },
        "feature_importance_hierarchy": importance_rankings
    }

@app.get("/dashboard")
def get_dashboard():
    """Serves the interactive web dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Dashboard index.html not found.")

@app.post("/predict")
def predict(inp: InputData):
    """
    Predicts mortality risk for an individual heart failure patient.
    Accepts payload format:
    {"data": {"age": 65, "ejection_fraction": 35, ...}}
    """
    if inp.data is None or not isinstance(inp.data, dict) or len(inp.data) == 0:
        raise HTTPException(
            status_code=422,
            detail="Input data payload cannot be empty. Please provide patient clinical parameters under the 'data' key."
        )

    try:
        result = compute_patient_prediction(inp.data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
def predict_batch(inp: BatchInputData):
    """
    Performs batch mortality risk predictions for a patient cohort.
    Returns individual patient scores and aggregate cohort risk statistics.
    """
    if not inp.patients:
        raise HTTPException(status_code=422, detail="Patient list cannot be empty.")

    results = []
    tier_counts = {"Very Low": 0, "Low": 0, "Moderate": 0, "High": 0, "Very High": 0}
    probabilities = []

    for idx, p in enumerate(inp.patients):
        pred = compute_patient_prediction(p)
        pred["patient_index"] = idx + 1
        tier_counts[pred["risk"]] = tier_counts.get(pred["risk"], 0) + 1
        probabilities.append(pred["probability"])
        results.append(pred)

    total = len(results)
    high_critical_count = tier_counts["High"] + tier_counts["Very High"]
    avg_prob = float(np.mean(probabilities)) if probabilities else 0.0

    return {
        "total_patients": total,
        "cohort_summary": {
            "average_mortality_probability": round(avg_prob, 4),
            "tier_distribution": tier_counts,
            "high_or_critical_percentage": f"{round((high_critical_count / total) * 100, 1)}%"
        },
        "predictions": results
    }

@app.post("/predict/explain")
def predict_explain(inp: InputData):
    """
    Provides in-depth patient risk factor attribution and personalized clinical advice.
    """
    if inp.data is None or not isinstance(inp.data, dict) or len(inp.data) == 0:
        raise HTTPException(status_code=422, detail="Input data payload cannot be empty.")

    pred = compute_patient_prediction(inp.data)
    drivers = pred["key_drivers"]

    return {
        "patient_assessment": pred,
        "clinical_drivers_count": len(drivers),
        "clinical_drivers": drivers,
        "clinical_management_advice": pred["action_protocol"]
    }
