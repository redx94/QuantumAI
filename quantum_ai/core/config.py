from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

class QuantumBackend(Enum):
    QISKIT = "qiskit"
    PENNYLANE = "pennylane"
    CIRQ = "cirq"
    BRAKET = "braket"

@dataclass
class QuantumAIConfig:
    backend: QuantumBackend
    use_real_hardware: bool = False
    quantum_device: Optional[str] = None
    security_level: int = 2
    enable_agi_features: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.value,
            "use_real_hardware": self.use_real_hardware,
            "quantum_device": self.quantum_device,
            "security_level": self.security_level,
            "enable_agi_features": self.enable_agi_features
        }
