from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

class QuantumLayer:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits)
        self.cr = ClassicalRegister(n_qubits)
        self.circuit = QuantumCircuit(self.qr, self.cr)
    
    def add_layer(self, params):
        """Add a parameterized quantum layer."""
        for i in range(self.n_qubits):
            self.circuit.rx(params[i], i)
            self.circuit.rz(params[i + self.n_qubits], i)
        
        for i in range(self.n_qubits - 1):
            self.circuit.cnot(i, i + 1)
    
    def measure(self):
        """Add measurement operations."""
        self.circuit.measure(self.qr, self.cr)
