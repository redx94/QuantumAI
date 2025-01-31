import numpy as np
from qiskit import Aer, execute
from qiskit.algorithms.optimizers import SPSA
from typing import List, Optional, Callable

class QuantumNeuralNetwork:
    def __init__(self, 
                 num_qubits: int,
                 num_layers: int,
                 optimizer: Optional[Callable] = None):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.optimizer = optimizer or SPSA(maxiter=100)
        self.parameters = np.random.randn(num_layers * num_qubits * 3)
        
    def create_circuit(self, input_data: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits)
        
        # Encode input data
        for i, x in enumerate(input_data):
            circuit.ry(x, i)
        
        # Add variational layers
        param_idx = 0
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                circuit.rx(self.parameters[param_idx], qubit)
                circuit.ry(self.parameters[param_idx+1], qubit)
                circuit.rz(self.parameters[param_idx+2], qubit)
                param_idx += 3
            
            # Add entangling layers
            for i in range(self.num_qubits-1):
                circuit.cx(i, i+1)
        
        return circuit

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        circuit = self.create_circuit(input_data)
        backend = Aer.get_backend('statevector_simulator')
        result = execute(circuit, backend).result()
        statevector = result.get_statevector()
        return np.abs(statevector)**2
