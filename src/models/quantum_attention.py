import torch
import torch.nn as nn
import pennylane as qml

class QFTAttentionHead(nn.Module):
    def __init__(self, n_qubits=5):
        super().__init__()
        self.n_qubits = n_qubits
        self.theta = nn.Parameter(torch.zeros(n_qubits))
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def quantum_op(inputs, params):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            qml.QFT(wires=range(n_qubits))
            for i in range(n_qubits):
                qml.RZ(params[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = quantum_op

    def forward(self, x):
        bs, seq_len, embed_dim = x.shape
        x_crop = x[:, :, :self.n_qubits].reshape(bs * seq_len, self.n_qubits)

        out_list = []
        for sample in x_crop:
            out_sample = self.qnode(sample, self.theta)
            out_list.append(out_sample)

        return torch.stack(out_list).reshape(bs, seq_len, self.n_qubits)
