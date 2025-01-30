
import pennylane as qml
import torch

class QuantumEmbedding:
    def __init__(self):
        self.dev = qml.device("default.qubit", wires=4)

    @qml.qnode(self.dev, interface='torch')
    def quantum_layer(self, inputs):
        qml.AngleEmbedding(inputs, wires=range(4))
        qml.StronglyEntanglingLayers(weights=None, wires=range(4))
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]

    def apply_quantum_logic(self, text):
        inputs = torch.tensor([ord(c) for c in text[:4]], dtype=torch.float32)
        quantum_output = self.quantum_layer(inputs)
        return f"Quantum-enhanced: {text} | {quantum_output}"