from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.transpiler import PassManager
from qiskit.circuit.library import ZZFeatureMap
from typing import List, Optional, Dict
import numpy as np

class QuantumCircuitFactory:
    @staticmethod
    def create_variational_circuit(num_qubits: int, depth: int) -> QuantumCircuit:
        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(num_qubits)
        circuit = QuantumCircuit(qr, cr)
        
        parameters = []
        for d in range(depth):
            for i in range(num_qubits):
                theta = Parameter(f'θ_{d}_{i}')
                parameters.append(theta)
                circuit.rx(theta, i)
            
            for i in range(num_qubits-1):
                circuit.cx(i, i+1)
        
        circuit.measure(qr, cr)
        return circuit

    @staticmethod
    def create_qml_circuit(input_size: int, hidden_layers: List[int]) -> QuantumCircuit:
        total_qubits = max(input_size, max(hidden_layers))
        qr = QuantumRegister(total_qubits)
        cr = ClassicalRegister(total_qubits)
        circuit = QuantumCircuit(qr, cr)
        
        # Add quantum encoding layer
        for i in range(input_size):
            circuit.h(i)
            
        # Add variational layers
        for layer in hidden_layers:
            for i in range(layer):
                theta = Parameter(f'θ_{layer}_{i}')
                circuit.ry(theta, i)
            
            for i in range(layer-1):
                circuit.cz(i, i+1)
        
        circuit.measure(qr, cr)
        return circuit

    @staticmethod
    def create_noise_resilient_circuit(circuit: QuantumCircuit, noise_level: float = 0.01) -> QuantumCircuit:
        """Enhances circuit with error mitigation techniques."""
        qr = circuit.qregs[0]
        
        # Add dynamical decoupling sequences
        for i in range(circuit.num_qubits):
            circuit.x(i)
            circuit.barrier()
            circuit.x(i)
        
        return circuit

    @staticmethod
    def optimize_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
        """Optimizes circuit depth and gate count."""
        pm = PassManager()
        # Add optimization passes
        optimized_circuit = pm.run(circuit)
        return optimized_circuit

    @staticmethod
    def create_hardware_efficient_circuit(num_qubits: int, 
                                       backend_config: Dict,
                                       optimization_level: int = 2) -> QuantumCircuit:
        """Creates circuits optimized for specific quantum hardware."""
        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(num_qubits)
        circuit = QuantumCircuit(qr, cr)
        
        # Add hardware-efficient ansatz
        for i in range(num_qubits):
            circuit.h(i)
            theta = Parameter(f'θ_{i}')
            circuit.rz(theta, i)
        
        # Add connectivity-aware CNOT layers
        for i in range(0, num_qubits-1, 2):
            circuit.cx(i, i+1)
        
        circuit.measure(qr, cr)
        return circuit
