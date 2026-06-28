import os

class TwinConfiguration:
    API_VERSION_PREFIX: str = "/api/v1"
    SERVICE_NAME: str = "EV Battery Digital Twin"
    MODEL_ARTIFACT_PATH: str = os.getenv("MODEL_ARTIFACT_PATH", "models/rul_cnn_v1.pt")
    ANOMALY_CONFIDENCE_THRESHOLD: float = 0.85

settings = TwinConfiguration()
