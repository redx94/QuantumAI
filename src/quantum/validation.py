from qiskit.test.mock import FakeBackend
from qiskit import execute, Aer
from typing import List, Tuple
import numpy as np

class CircuitValidator:
    @staticmethod
    def validate_circuit(circuit, expected_state: List[complex]) -> bool:
        """Validates circuit against expected quantum state."""
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circuit, backend)
        state_vector = job.result().get_statevector()
        fidelity = np.abs(np.dot(state_vector, np.conj(expected_state)))
        return fidelity > 0.99

    @staticmethod
    def test_noise_resilience(circuit, shots: int = 1000) -> Tuple[float, float]:
        """Tests circuit performance with and without noise."""
        ideal_backend = Aer.get_backend('qasm_simulator')
        noisy_backend = FakeBackend()
        
        ideal_results = execute(circuit, ideal_backend, shots=shots).result()
        noisy_results = execute(circuit, noisy_backend, shots=shots).result()
        
        return ideal_results.get_counts(), noisy_results.get_counts()

    @staticmethod
    def verify_circuit_depth(circuit, max_depth: int) -> bool:
        """Verifies if circuit meets depth requirements."""
        return circuit.depth() <= max_depth
