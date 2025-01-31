from qiskit.result import Result
from qiskit import execute, Aer
import numpy as np

class QuantumAnalyzer:
    @staticmethod
    def run_with_error_analysis(circuit, shots=1000, noise_model=None):
        """Execute circuit and analyze results with error statistics"""
        backend = Aer.get_backend('qasm_simulator')
        
        result = execute(
            circuit,
            backend=backend,
            noise_model=noise_model,
            shots=shots
        ).result()
        
        counts = result.get_counts()
        # Calculate error rates and fidelity
        error_rate = 1 - max(counts.values()) / shots
        
        return {
            'counts': counts,
            'error_rate': error_rate,
            'raw_result': result
        }
    
    @staticmethod
    def calculate_state_fidelity(result: Result, target_state):
        """Calculate fidelity between result and target state"""
        state_vector = result.get_statevector()
        fidelity = np.abs(np.dot(np.conjugate(state_vector), target_state))**2
        return fidelity
