import torch
import torch.nn as nn
from ..quantum.circuit import QuantumLayer

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.quantum_layer = QuantumLayer(n_qubits)
        self.params = nn.Parameter(torch.randn(n_layers, 2 * n_qubits))
        
    def forward(self, x):
        batch_size = x.shape[0]
        output = []
        
        for batch in range(batch_size):
            for layer in range(self.n_layers):
                self.quantum_layer.add_layer(self.params[layer])
            output.append(self.quantum_layer.measure())
            
        return torch.tensor(output)
