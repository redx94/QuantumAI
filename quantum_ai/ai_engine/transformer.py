import torch
import torch.nn as nn
from typing import Optional

from quantum_ai.q_fabric.hybrid_layer import QuantumLayer
from quantum_ai.ai_engine.core import BaseAIModel

class QuantumTransformerBlock(BaseAIModel):
    """Transformer block with quantum-enhanced attention."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_qubits: int = 4
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.quantum_layer = QuantumLayer(n_qubits, n_layers=2)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantum-enhanced attention."""
        # Standard attention
        attended = self.attention(x, x, x)[0]
        x = self.norm1(x + attended)
        
        # Quantum enhancement
        q_enhanced = self.quantum_layer(x.view(-1, self.quantum_layer.n_qubits))
        q_enhanced = q_enhanced.view(x.shape)
        
        # Feed-forward
        ff_out = self.feed_forward(x + q_enhanced)
        return self.norm2(x + ff_out)
