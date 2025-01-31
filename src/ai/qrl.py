import numpy as np
from ..quantum.core import QuantumCompute
import torch

class QuantumQLearning:
    def __init__(self, state_dim: int, action_dim: int, n_qubits: int = 8):
        self.quantum = QuantumCompute(n_qubits=n_qubits)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def select_action(self, state: np.ndarray) -> int:
        quantum_values = []
        for a in range(self.action_dim):
            action_params = np.zeros(self.action_dim)
            action_params[a] = 1.0
            q_value = self.quantum.qrl_step(state, action_params)
            quantum_values.append(q_value)
            
        return np.argmax(quantum_values)
        
    def update(self, state, action, reward, next_state):
        current_q = self.quantum.qrl_step(state, np.eye(self.action_dim)[action])
        next_max_q = max(self.quantum.qrl_step(next_state, np.eye(self.action_dim)[a]) 
                        for a in range(self.action_dim))
        
        # Quantum Bellman update
        loss = reward + 0.99 * next_max_q - current_q
        return loss.item()
