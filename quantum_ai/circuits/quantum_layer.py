import pennylane as qml
import torch
from torch import nn

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Initialize trainable parameters
        self.params = nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        
    @qml.qnode(device=dev)
    def quantum_circuit(self, inputs, params):
        # Encode inputs
        for i in range(self.n_qubits):
            qml.RX(inputs[i], wires=i)
            
        # Apply variational layers
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                qml.Rot(*params[layer, qubit], wires=qubit)
            for q1 in range(self.n_qubits - 1):
                qml.CNOT(wires=[q1, q1 + 1])
                
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x):
        batch_size = x.shape[0]
        expectations = torch.zeros(batch_size, self.n_qubits)
        
        for b in range(batch_size):
            expectations[b] = torch.tensor(
                self.quantum_circuit(x[b], self.params)
            )
        return expectations
