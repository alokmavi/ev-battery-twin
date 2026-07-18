# EV Battery Digital Twin

Predictive Maintenance microservice for estimating Remaining Useful Life (RUL) of EV batteries using a 1D Convolutional Neural Network.

## Architecture
- **Inference Engine:** PyTorch (1D CNN for time-series extraction)
- **Routing Layer:** FastAPI
- **Validation:** Pydantic

Architecture Overview
This repository implements a predictive digital twin for lithium-ion battery arrays. It processes sequential charge/discharge telemetry (voltage, current, temperature) through a 1D Convolutional Neural Network designed for time-series feature extraction. The model is wrapped in an asynchronous FastAPI routing layer with strict Pydantic payload validation, ensuring fault tolerance and memory-safe tensor inference for edge-deployment environments.

## Quickstart

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    uvicorn src.api:app --reload
