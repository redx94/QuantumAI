import pennylane as qml
import torch
from torch import nn
from .error_mitigation import NoiseAwareQuantumLayer

class QuantumLayer(NoiseAwareQuantumLayer):
    """Quantum layer with built-in error mitigation capabilities."""
    
    def __init__(self, n_qubits: int, n_layers: int):
        """Initialize quantum layer with error mitigation enabled by default."""
        super().__init__(n_qubits, n_layers)
