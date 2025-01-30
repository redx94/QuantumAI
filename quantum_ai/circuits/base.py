from abc import ABC, abstractmethod
import pennylane as qml
import qiskit

class QuantumCircuit(ABC):
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.device = qml.device('default.qubit', wires=n_qubits)

    @abstractmethod
    def build_circuit(self):
        pass

    @abstractmethod
    def run(self, inputs):
        pass

class VQC(QuantumCircuit):
    def build_circuit(self):
        @qml.qnode(self.device)
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit
