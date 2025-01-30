from abc import ABC, abstractmethod
from typing import Dict, Any

class QuantumProvider(ABC):
    @abstractmethod
    async def execute_circuit(self, circuit: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a quantum circuit on the provider's QPU"""
        pass

    @abstractmethod
    async def get_backend_metrics(self) -> Dict[str, float]:
        """Get current metrics (error rates, queue time, cost) for the QPU"""
        pass
