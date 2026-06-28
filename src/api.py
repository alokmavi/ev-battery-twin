from fastapi import FastAPI, HTTPException
from src.config import settings
from src.schemas import BatteryTelemetryPayload, RemainingUsefulLifePrediction
from src.inference import PredictionEngine
import torch
import numpy as np

app = FastAPI(title=settings.SERVICE_NAME, openapi_url=f"{settings.API_VERSION_PREFIX}/openapi.json")
inference_engine = PredictionEngine()

@app.get("/healthz")
async def check_readiness():
    return {"status": "operational", "engine": "cnn_rul_predictor"}

@app.post(f"{settings.API_VERSION_PREFIX}/predict", response_model=RemainingUsefulLifePrediction)
async def process_telemetry(payload: BatteryTelemetryPayload):
    try:
        mock_sequence = np.random.normal(
            loc=[payload.voltage_discharge_v, payload.current_discharge_a, payload.temperature_celsius],
            scale=[0.05, 0.1, 0.5],
            size=(50, 3)
        )
        input_tensor = torch.tensor(mock_sequence, dtype=torch.float32)
        
        rul_cycles = inference_engine.estimate_remaining_life(input_tensor)
        
        return RemainingUsefulLifePrediction(
            battery_id=payload.battery_id,
            predicted_rul_cycles=max(0, int(rul_cycles)),
            confidence_interval=0.92,
            anomalous_thermal_state=payload.temperature_celsius > 45.0
        )
    except Exception as prediction_fault:
        raise HTTPException(status_code=500, detail=str(prediction_fault))
