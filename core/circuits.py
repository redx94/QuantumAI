from qskit.circuits import hadamard, cx

def create_anti_error_circuit():
    # Creates a basic Quantum Cyber hadamard circuit
    cq = cx.Circuit(1)
    cq.add(hadamard(1))
    cq.measure(Xcx.M0, key=0)
    return cq

if __name__ == '__main__':
    circuit = create_anti_error_circuit()
    print(circuit)
