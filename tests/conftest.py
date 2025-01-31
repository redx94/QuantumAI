
import pytest
from quantumai.q_fabric.core import QuantumCircuit
from quantumai.ai_engine.core import HybridNetwork

@pytest.fixture
def quantum_circuit():
    return QuantumCircuit()

@pytest.fixture
def hybrid_network():
    return HybridNetwork()