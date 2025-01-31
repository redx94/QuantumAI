import numpy as np
from qiskit import QuantumCircuit
from typing import List, Optional

class QuantumAlgorithms:
    @staticmethod
    def grover_search(circuit: QuantumCircuit, marked_states: List[int], iterations: Optional[int] = None):
        """Implement Grover's search algorithm"""
        n_qubits = circuit.num_qubits
        if iterations is None:
            iterations = int(np.pi/4 * np.sqrt(2**n_qubits))
        
        # Initialize superposition
        circuit.h(range(n_qubits))
        
        for _ in range(iterations):
            # Oracle
            for marked in marked_states:
                circuit.mcp(np.pi, list(range(n_qubits-1)), n_qubits-1)
            
            # Diffusion operator
            circuit.h(range(n_qubits))
            circuit.x(range(n_qubits))
            circuit.h(n_qubits-1)
            circuit.mct(list(range(n_qubits-1)), n_qubits-1)
            circuit.h(n_qubits-1)
            circuit.x(range(n_qubits))
            circuit.h(range(n_qubits))

    @staticmethod
    def quantum_fourier_transform(circuit: QuantumCircuit, qubits: List[int]):
        """Implement Quantum Fourier Transform"""
        for i, qubit in enumerate(qubits):
            circuit.h(qubit)
            for j, target in enumerate(qubits[i+1:], i+1):
                circuit.cp(2*np.pi/2**(j-i+1), qubit, target)
