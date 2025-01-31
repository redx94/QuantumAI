from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy

class QuantumCloudInterface:
    def __init__(self, api_token=None):
        self.api_token = api_token
        self.provider = None

    def connect(self):
        """
        Connect to IBMQ
        """
        IBMQ.save_account(self.api_token)
        self.provider = IBMQ.load_account()

    def get_backend(self, min_qubits=5):
        """
        Get the least busy backend with minimum required qubits
        """
        available_backends = self.provider.backends(
            filters=lambda x: x.configuration().n_qubits >= min_qubits and 
                            not x.configuration().simulator and 
                            x.status().operational
        )
        return least_busy(available_backends)

    async def run_job(self, circuit, shots=1000):
        """
        Run quantum circuit on real quantum computer
        """
        backend = self.get_backend()
        job = await backend.run(circuit, shots=shots)
        return job
