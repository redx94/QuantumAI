
import qiskit
import pennylane as qml

def execute_on_qiskit(circuit):
    backend = qiskit.Aer.get_backend('qasm_simulator')
    job = qiskit.execute(circuit, backend)
    result = job.result()
    return result.get_counts()

def execute_on_pennylane(circuit):
    dev = qml.device("default.qubit", wires=circuit.num_wires)
    @qml.qnode(dev)
    def qnode(inputs):
        circuit(inputs)
        return [qml.expval(qml.PauliZ(i)) for i in range(circuit.num_wires)]
    return qnode