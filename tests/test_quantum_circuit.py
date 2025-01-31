import pytest
from quantumai.quantum.circuit import QuantumLayer
import numpy as np

def test_quantum_layer_initialization():
    n_qubits = 2
    layer = QuantumLayer(n_qubits)
    assert layer.n_qubits == n_qubits
    assert layer.circuit is not None

def test_add_layer():
    layer = QuantumLayer(2)
    params = np.random.random(4)
    layer.add_layer(params)
    # Assert circuit depth is as expected
    assert layer.circuit.depth() > 0
