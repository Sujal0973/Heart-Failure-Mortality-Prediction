"""
Automated Test Suite for Heart Failure Mortality Prediction API
"""

import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data
    assert "features" in data
    assert len(data["features"]) == 12
    assert "docs_url" in data
    assert "health_url" in data

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["scaler_loaded"] is True
    assert data["feature_count"] == 12

def test_features_endpoint():
    response = client.get("/features")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert len(data["features"]) == 12
    # Verify mandatory feature keys
    for f in data["features"]:
        assert "name" in f
        assert "label" in f
        assert "unit" in f
        assert "normal_range" in f
        assert "importance_weight" in f

def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "XGBoost" in data["algorithm"]
    assert "hyperparameters" in data
    assert "evaluation_metrics" in data
    assert "feature_importance_hierarchy" in data
    assert len(data["feature_importance_hierarchy"]) == 12

def test_predict_low_risk_patient():
    low_risk_payload = {
        "data": {
            "age": 45,
            "anaemia": 0,
            "creatinine_phosphokinase": 120,
            "diabetes": 0,
            "ejection_fraction": 60,
            "high_blood_pressure": 0,
            "platelets": 300000,
            "serum_creatinine": 0.8,
            "serum_sodium": 140,
            "sex": 1,
            "smoking": 0,
            "time": 250
        }
    }
    response = client.post("/predict", json=low_risk_payload)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert "risk" in data
    assert data["probability"] < 0.20
    assert data["risk"] in ["Very Low", "Low"]
    assert "risk_color" in data
    assert "action_protocol" in data

def test_predict_high_risk_patient():
    high_risk_payload = {
        "data": {
            "age": 85,
            "anaemia": 1,
            "creatinine_phosphokinase": 4500,
            "diabetes": 1,
            "ejection_fraction": 18,
            "high_blood_pressure": 1,
            "platelets": 110000,
            "serum_creatinine": 3.8,
            "serum_sodium": 125,
            "sex": 1,
            "smoking": 1,
            "time": 15
        }
    }
    response = client.post("/predict", json=high_risk_payload)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert "risk" in data
    assert data["probability"] > 0.60
    assert data["risk"] in ["High", "Very High"]
    assert len(data["key_drivers"]) >= 3

def test_predict_empty_payload_validation():
    response = client.post("/predict", json={"data": {}})
    assert response.status_code == 422

def test_predict_missing_data_key():
    response = client.post("/predict", json={})
    assert response.status_code == 422

def test_predict_batch():
    batch_payload = {
        "patients": [
            {
                "age": 45, "anaemia": 0, "creatinine_phosphokinase": 200, "diabetes": 0,
                "ejection_fraction": 60, "high_blood_pressure": 0, "platelets": 300000,
                "serum_creatinine": 0.9, "serum_sodium": 140, "sex": 1, "smoking": 0, "time": 250
            },
            {
                "age": 80, "anaemia": 1, "creatinine_phosphokinase": 3000, "diabetes": 1,
                "ejection_fraction": 20, "high_blood_pressure": 1, "platelets": 150000,
                "serum_creatinine": 3.0, "serum_sodium": 126, "sex": 0, "smoking": 0, "time": 20
            }
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_patients"] == 2
    assert "cohort_summary" in data
    assert "average_mortality_probability" in data["cohort_summary"]
    assert len(data["predictions"]) == 2

def test_predict_explain():
    payload = {
        "data": {
            "age": 75,
            "anaemia": 1,
            "creatinine_phosphokinase": 600,
            "diabetes": 0,
            "ejection_fraction": 25,
            "high_blood_pressure": 1,
            "platelets": 250000,
            "serum_creatinine": 2.2,
            "serum_sodium": 130,
            "sex": 1,
            "smoking": 1,
            "time": 25
        }
    }
    response = client.post("/predict/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "patient_assessment" in data
    assert "clinical_drivers" in data
    assert data["clinical_drivers_count"] > 0
    assert "clinical_management_advice" in data

def test_dashboard_endpoint():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Heart Failure" in response.text
