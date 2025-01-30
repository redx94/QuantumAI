import numpy as np
from ..quantum.core import QuantumCompute
from ..quantum.embeddings import QuantumEmbedding

class QuantumRL:
    def __init__(self):
        self.quantum = QuantumCompute()
        self.embedding = QuantumEmbedding()
        
    def quantum_policy(self, state):
        """Convert state to quantum policy using embeddings"""
        quantum_state = self.embedding.encode(state)
        return np.array(quantum_state)
    
    def train_step(self, state, reward):
        """Single step of quantum RL training"""
        policy = self.quantum_policy(state)
        # Update policy based on reward
        return policy
