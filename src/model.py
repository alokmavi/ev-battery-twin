import torch
import torch.nn as nn
import torch.nn.functional as F

class RULPredictorCNN(nn.Module):
    def __init__(self, sequence_length: int = 50, feature_dimensions: int = 3):
        super().__init__()
        self.conv_layer_1 = nn.Conv1d(in_channels=feature_dimensions, out_channels=64, kernel_size=3, padding=1)
        self.conv_layer_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        
        self.fc_layer_1 = nn.Linear(128 * (sequence_length // 2), 64)
        self.fc_layer_2 = nn.Linear(64, 1)

    def forward(self, telemetry_tensor: torch.Tensor) -> torch.Tensor:
        tensor_permuted = telemetry_tensor.permute(0, 2, 1)
        
        activation_1 = F.relu(self.conv_layer_1(tensor_permuted))
        activation_2 = self.max_pool(F.relu(self.conv_layer_2(activation_1)))
        
        flattened_tensor = torch.flatten(activation_2, 1)
        dense_out = F.relu(self.fc_layer_1(flattened_tensor))
        predicted_rul = self.fc_layer_2(dense_out)
        
        return predicted_rul
