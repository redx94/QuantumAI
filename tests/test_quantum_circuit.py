import pytest


from quantumai.quantum.circuit import QuantumLayer
import numpy as np
from unittest.mock import MagicMock

def test_quantum_layer_initialization():
    n_qubits = 2
    layer = QuantumLayer(n_qubits)
    assert layer.n_qubits == n_qubits
    assert layer.circuit is not None

def test_add_layer():
    layer = QuantumLayer(2)
    params = np.random.random(4)
    layer.add_layer(params)
    assert layer.circuit.depth() > 0

@pytest.fixture
def basic_layer():
    return QuantumLayer(2)

def test_parameter_validation(basic_layer):
    with pytest.raises(ValueError):
        basic_layer.add_layer(np.random.random(3))
    basic_layer.add_layer(np.random.random(4))

def test_measurement_outcomes(basic_layer):
    params = np.array([np.pi/2, 0, np.pi/2, 0])
    basic_layer.add_layer(params)
    probs = basic_layer.measure()
    expected_probs = np.array([0.5, 0.5, 0, 0])
    np.testing.assert_allclose(probs, expected_probs, atol=1e-3)

def test_invalid_qubit_count():
    with pytest.raises(ValueError):
        QuantumLayer(0)

def test_error_mitigation_techniques():
    layer = QuantumLayer(2)
    layer.enable_error_mitigation(shots=1000)
    params = np.array([np.pi/2, 0, np.pi/2, 0])
    layer.add_layer(params)
    probs = layer.measure()
    assert np.isclose(np.sum(probs), 1.0, atol=0.1)
    assert hasattr(layer, 'measurement_filters')
    assert layer.mitigation_techniques['meas_calib']
    assert layer.mitigation_techniques['zne']

def test_backend_execution():
    for backend in ['qasm_simulator', 'statevector_simulator', 'unitary_simulator']:
        layer = QuantumLayer(2, backend=backend)
        params = np.random.random(4)
        layer.add_layer(params)
        probs = layer.measure()
        assert len(probs) ==
