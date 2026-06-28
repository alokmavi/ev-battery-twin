import torch
from src.model import RULPredictorCNN
from src.config import settings
import logging

class PredictionEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RULPredictorCNN().to(self.device)
        self._load_weights()
        self.model.eval()

    def _load_weights(self) -> None:
        try:
            state_dict = torch.load(settings.MODEL_ARTIFACT_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logging.info(f"Successfully mounted weights from {settings.MODEL_ARTIFACT_PATH}")
        except FileNotFoundError:
            # Fallback for development environments without synced blob storage
            logging.warning("No pretrained weights located. Engine initializing with random initialization.")

    @torch.no_grad()
    def estimate_remaining_life(self, sequence_tensor: torch.Tensor) -> float:
        tensor_device = sequence_tensor.to(self.device)
        
        if tensor_device.dim() == 2:
            tensor_device = tensor_device.unsqueeze(0)
            
        rul_prediction = self.model(tensor_device)
        return float(rul_prediction.item())
