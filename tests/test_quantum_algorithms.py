import unittest
import numpy as np
from src.quantum_algorithms import AdvancedQuantumAlgorithms
from src.quantum_circuit import QuantumCircuitEnhanced

class TestQuantumAlgorithms(unittest.TestCase):
    def setUp(self):
        self.algorithms = AdvancedQuantumAlgorithms()

    def test_grover_algorithm(self):
        # Test Grover's algorithm
        n_qubits = 3
        oracle = QuantumCircuit(n_qubits)
        circuit = self.algorithms.grover_algorithm(oracle, n_qubits)
        self.assertEqual(circuit.num_qubits, n_qubits)

    def test_qft(self):
        # Test Quantum Fourier Transform
        n_qubits = 3
        circuit = self.algorithms.quantum_fourier_transform(n_qubits)
        self.assertEqual(circuit.num_qubits, n_qubits)

    def test_circuit_visualization(self):
        # Test circuit visualization
        circuit = QuantumCircuitEnhanced(2)
        visualization = circuit.visualize()
        self.assertIsNotNone(visualization)
