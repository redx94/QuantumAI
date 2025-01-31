import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import circuit_drawer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CXCancellation

class QuantumCircuitEnhanced:
    def __init__(self, num_qubits):
        self.circuit = QuantumCircuit(num_qubits)
        self.num_qubits = num_qubits
        self.noise_model = None
        self.optimization_level = 1

    def visualize(self, output_format='mpl'):
        """
        Visualize the quantum circuit
        Args:
            output_format (str): 'mpl', 'text', or 'latex'
        """
        return circuit_drawer(self.circuit, output=output_format)

    def apply_error_correction(self):
        """
        Implements basic quantum error correction using the 3-qubit code
        """
        # Add error correction code
        self.circuit.h(0)
        self.circuit.cx(0, 1)
        self.circuit.cx(0, 2)

    def set_noise_model(self, noise_params):
        """Add custom noise model to the circuit"""
        noise_model = NoiseModel()
        # Add thermal relaxation
        if 'T1' in noise_params and 'T2' in noise_params:
            for qubit in range(self.num_qubits):
                noise_model.add_quantum_error(
                    thermal_relaxation_error(
                        noise_params['T1'],
                        noise_params['T2'],
                        noise_params.get('gate_time', 0.1)
                    ),
                    ['u1', 'u2', 'u3'], [qubit]
                )
        self.noise_model = noise_model

    def optimize_circuit(self, level=None):
        """Optimize the quantum circuit"""
        if level is not None:
            self.optimization_level = level
        
        pm = PassManager()
        pm.append(Optimize1qGates())
        pm.append(CXCancellation())
        self.circuit = pm.run(self.circuit)

    def add_error_syndrome_measurement(self):
        """Add syndrome measurements for error correction"""
        cr = ClassicalRegister(self.num_qubits)
        self.circuit.add_register(cr)
        for i in range(self.num_qubits-1):
            self.circuit.cx(i, i+1)
            self.circuit.measure(i, i)
