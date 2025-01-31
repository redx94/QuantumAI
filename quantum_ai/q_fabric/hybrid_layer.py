from typing import Optional, Tuple

import pennylane as qml
import torch
import torch.nn as nn

class QuantumLayer(nn.Module):
    """Quantum circuit as a PyTorch layer."""
    
    def __init__(
        self,
        n_qubits: int,
        n_layers: int,
        backend: str = "default.qubit"
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(backend, wires=n_qubits)
        
        # Initialize trainable parameters
        self.theta = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute quantum circuit with classical input."""
        batch_size = x.shape[0]
        
        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights):
            # Encode classical data
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            
            # Apply variational layers
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    qml.Rot(*weights[layer, qubit], wires=qubit)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        results = []
        for i in range(batch_size):
            results.append(quantum_circuit(x[i], self.theta))
        
        return torch.stack(results)
