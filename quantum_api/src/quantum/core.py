from qiskit import QuantumCircuit, Aer, execute

class QuantumCompute:
    def __init__(self):
        self.backend = Aer.get_backend('qasm_simulator')
    
    def create_bell_state(self):
        """Creates a simple Bell state"""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        return circuit
    
    def run_circuit(self, circuit):
        """Executes a quantum circuit"""
        job = execute(circuit, self.backend, shots=1000)
        return job.result().get_counts()
