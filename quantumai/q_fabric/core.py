from abc import ABC, abstractmethod
from typing import Any, List, Optional

class QuantumBackend(ABC):
    @abstractmethod
    def execute_circuit(self, circuit: Any) -> Any:
        pass

class QuantumCircuit:
    def __init__(self, backend: Optional[QuantumBackend] = None):
        self.backend = backend or DefaultSimulatorBackend()
        self.gates: List[Any] = []

    def add_gate(self, gate: Any) -> None:
        self.gates.append(gate)

    def execute(self) -> Any:
        return self.backend.execute_circuit(self)

class DefaultSimulatorBackend(QuantumBackend):
    def execute_circuit(self, circuit: Any) -> Any:
        # Default simulator implementation
        return NotImplemented
