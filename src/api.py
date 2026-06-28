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
