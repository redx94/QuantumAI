import pennylane as qml
import numpy as np

class QuantumEmbedding:
    def __init__(self, n_qubits=4):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
    @qml.qnode(lambda self: self.dev)
    def encode(self, inputs):
        # Encode classical data into quantum state
        qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
        # Apply quantum transformation
        qml.StronglyEntanglingLayers(
            weights=np.random.randn(3, self.n_qubits, 3),
            wires=range(self.n_qubits)
        )
        # Measure in computational basis
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
