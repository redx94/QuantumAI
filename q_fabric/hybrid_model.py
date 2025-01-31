import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface='tf')
def quantum_circuit(inputs):
    for i in range(n_qubits):
        qml.RX(inputs[i], wires=i)
    qml.CZ(wires=[0, 1])
    qml.CZ(wires=[2, 3])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridQuantumModel(tf.keras.Model):
    def __init__(self):
        super(HybridQuantumModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(8, activation='relu')
        self.dense2 = tf.keras.layers.Dense(n_qubits, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return quantum_circuit(x)
