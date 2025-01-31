from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader

class QuantumAITrainer:
    """Training manager for quantum-classical hybrid models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
    def train_epoch(
        self,
        dataloader: DataLoader,
        quantum_enabled: bool = True
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return {"train_loss": total_loss / len(dataloader)}
