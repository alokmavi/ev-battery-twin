import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TwinTrainer:
    def __init__(self, model: nn.Module, learning_rate: float = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader, epoch_idx: int) -> float:
        self.model.train()
        cumulative_epoch_loss = 0.0
        
        for batch_idx, (features, targets) in enumerate(dataloader):
            features, targets = features.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(features)
            
            loss = self.criterion(predictions, targets)
            loss.backward()
            self.optimizer.step()
            
            cumulative_epoch_loss += loss.item()
            
        average_loss = cumulative_epoch_loss / len(dataloader)
        logging.info(f"Epoch {epoch_idx} completed. Average MSE Loss: {average_loss:.4f}")
        return average_loss
