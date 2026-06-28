from fastapi.testclient import TestClient
from src.api import app
from src.config import settings

client = TestClient(app)

def test_readiness_probe():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "operational"

def test_predict_endpoint_validation():
    payload = {
        "battery_id": "BAT-99",
        "cycle_index": 0,
        "voltage_discharge_v": 3.8,
        "current_discharge_a": -1.5,
        "temperature_celsius": 25.0
    }
    response = client.post(f"{settings.API_VERSION_PREFIX}/predict", json=payload)
    assert response.status_code == 422
