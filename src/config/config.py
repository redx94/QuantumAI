from dataclasses import dataclass
from typing import Dict, Any
import yaml

@dataclass
class QuantumConfig:
    num_qubits: int
    noise_model: str
    optimization_level: int
    backend_name: str

class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_quantum_config(self) -> QuantumConfig:
        qc = self.config.get('quantum', {})
        return QuantumConfig(
            num_qubits=qc.get('num_qubits', 5),
            noise_model=qc.get('noise_model', 'ideal'),
            optimization_level=qc.get('optimization_level', 1),
            backend_name=qc.get('backend_name', 'qasm_simulator')
        )
