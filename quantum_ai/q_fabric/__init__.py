"""
Quantum Fabric Layer
------------------
Core quantum computing abstractions and implementations.
"""

from enum import Enum

class QuantumBackend(Enum):
    QISKIT = "qiskit"
    PENNYLANE = "pennylane"
    CIRQ = "cirq"
    BRAKET = "braket"