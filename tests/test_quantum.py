import pytest
from quantum.core import QuantumCompute
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
