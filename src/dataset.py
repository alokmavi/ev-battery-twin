import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple

class CycleTelemetryDataset(Dataset):
    def __init__(self, sequence_length: int = 50, feature_dimensions: int = 3):
        self.sequence_length = sequence_length
        self.feature_dimensions = feature_dimensions
        # Simulating NASA battery dataset characteristics for edge testing
        self.mock_capacity_degradation = np.linspace(2.0, 1.4, 200)

    def __len__(self) -> int:
        return len(self.mock_capacity_degradation) - self.sequence_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            telemetry_sequence = np.random.normal(
                loc=[3.8, -1.5, 25.0], 
                scale=[0.1, 0.5, 2.0], 
                size=(self.sequence_length, self.feature_dimensions)
            )
            
            remaining_cycles = len(self.mock_capacity_degradation) - (index + self.sequence_length)
            
            return (
                torch.tensor(telemetry_sequence, dtype=torch.float32),
                torch.tensor([remaining_cycles], dtype=torch.float32)
            )
        except Exception as buffer_fault:
            raise RuntimeError(f"Failed to load telemetry sequence at index {index}") from buffer_fault
