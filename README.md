# EV Battery Digital Twin

Predictive Maintenance microservice for estimating Remaining Useful Life (RUL) of EV batteries using a 1D Convolutional Neural Network.

## Architecture
- **Inference Engine:** PyTorch (1D CNN for time-series extraction)
- **Routing Layer:** FastAPI
- **Validation:** Pydantic

## Quickstart

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    uvicorn src.api:app --reload
