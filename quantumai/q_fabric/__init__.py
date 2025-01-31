"""Quantum Fabric: Hardware abstraction and optimization layer."""

from enum import Enum

class QuantumBackend(Enum):
    """Supported quantum computing backends."""
    QISKIT = "qiskit"
    PENNYLANE = "pennylane"
    CIRQ = "cirq"
    BRAKET = "braket"

class QuantumOptimizer(Enum):
    QAOA = "qaoa"
    VQE = "vqe"
    QUANTUM_NATURAL_GRADIENT = "qng"
