from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class QuantumCircuit(ABC):
    """Abstract base class for quantum circuits."""
    
    def __init__(self, n_qubits: int, backend: str):
        self.n_qubits = n_qubits
        self.backend = backend
        self._circuit: Any = None
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the quantum circuit."""
        pass
    
    @abstractmethod
    def add_layer(self, params: List[float]) -> None:
        """Add a parameterized quantum layer."""
        pass
    
    @abstractmethod
    def measure(self) -> Dict[str, float]:
        """Execute measurement and return results."""
        pass

class QuantumOptimizationCircuit(QuantumCircuit):
    """Specialized circuit for quantum optimization tasks."""
    
    def __init__(self, n_qubits: int, backend: str, optimizer_type: str):
        super().__init__(n_qubits, backend)
        self.optimizer_type = optimizer_type
        self.optimization_params: Dict[str, Any] = {}
