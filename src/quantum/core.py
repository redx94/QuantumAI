import pennylane as qml
import numpy as np
from typing import Dict, List, Union

class QuantumCompute:
    def __init__(self):
        """Initialize quantum device using PennyLane's default.qubit"""
        self.dev = qml.device("default.qubit", wires=2)
    
    @qml.qnode(device=qml.device("default.qubit", wires=2))
    def create_bell_state(self) -> List[float]:
        """Creates a Bell state using PennyLane"""
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
    
    def run_circuit(self, measurements: List[float]) -> Dict[str, float]:
        """Process quantum measurements"""
        return {
            "z0": measurements[0],
            "z1": measurements[1],
            "correlation": measurements[0] * measurements[1]
        }
    
    def quantum_embedding(self, input_vector: List[float]) -> np.ndarray:
        """Encodes input into quantum state using PennyLane"""
        dev = qml.device("default.qubit", wires=len(input_vector))
        
        @qml.qnode(dev)
        def circuit(inputs):
            qml.AngleEmbedding(inputs, wires=range(len(inputs)))
            qml.StronglyEntanglingLayers(
                weights=np.random.rand(3, len(inputs), 3), 
                wires=range(len(inputs))
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(len(inputs))]
        
        return circuit(input_vector)
