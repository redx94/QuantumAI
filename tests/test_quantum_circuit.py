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

@pytest.fixture
def basic_layer():
    return QuantumLayer(2)

def test_parameter_validation(basic_layer):
    """Test parameter validation logic"""
    with pytest.raises(ValueError):
        # Incorrect parameter count (needs 4 params for 2 qubits)
        basic_layer.add_layer(np.random.random(3))
    
    # Valid parameters should not raise
    basic_layer.add_layer(np.random.random(4))

def test_measurement_outcomes(basic_layer):
    """Test quantum measurement probabilities with fixed parameters"""
    # Fixed parameters for predictable rotations
    params = np.array([np.pi/2, 0, np.pi/2, 0])  # Ry(pi/2) on both qubits
    basic_layer.add_layer(params)
    
    # Simulate circuit
    probs = basic_layer.measure()
    
    # Expected probabilities after Ry(pi/2) gates
    expected_probs = np.array([0.5, 0.5, 0, 0])  # Equal superposition
    np.testing.assert_allclose(probs, expected_probs, atol=1e-3)

def test_invalid_qubit_count():
    """Test error handling for invalid qubit counts"""
    with pytest.raises(ValueError):
        QuantumLayer(0)  # Invalid qubit count
