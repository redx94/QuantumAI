"""AI Engine: Hybrid quantum-classical AI models and AGI components."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class QuantumAIConfig:
    backend: str
    n_qubits: int
    n_layers: int
    classical_units: Optional[list[int]] = None
