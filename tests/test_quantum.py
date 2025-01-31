import pytest
from quantum_ai.quantum.core import QuantumCore
import numpy as np
import pennylane as qml

def test_quantum_execution():
    qc = QuantumCompute()
    
    def test_circuit():
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))
        
    result = qc.execute(test_circuit)
    assert isinstance(result, float)

def test_batch_execution():
    qc = QuantumCompute()
    circuits = [test_circuit for _ in range(3)]
    results = qc.batch_execute(circuits)
    assert len(results) == 3

def test_quantum_core():
    qc = QuantumCore(n_qubits=2)
    inputs = np.random.randn(2)
    weights = np.random.randn(10)
    
    result = qc.quantum_layer(inputs, weights)
    assert isinstance(result, float)
    assert -1 <= result <= 1

def test_quantum_circuit_creation():
    qc = QuantumCore(n_qubits=2)
    assert qc.circuit.num_qubits == 2
