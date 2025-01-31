from abc import ABC, abstractmethod
import qiskit
import cirq
import pyquil
from typing import Any, Dict

class QPUProvider(ABC):
    @abstractmethod
    def get_backend(self) -> Any:
        pass
    
    @abstractmethod
    def execute_circuit(self, circuit: Any, **kwargs) -> Any:
        pass

class IBMProvider(QPUProvider):
    def __init__(self, token: str):
        self.provider = qiskit.IBMQ.enable_account(token)
        
    def get_backend(self, name: str = 'ibmq_lima'):
        return self.provider.get_backend(name)
        
    def execute_circuit(self, circuit, shots=1000):
        backend = self.get_backend()
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result()

class GoogleProvider(QPUProvider):
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.sampler = cirq.Sampler()
        
    def get_backend(self):
        return self.sampler
        
    def execute_circuit(self, circuit, repetitions=1000):
        return self.sampler.run(circuit, repetitions=repetitions)

class RigettiProvider(QPUProvider):
    def __init__(self, api_key: str):
        self.qvm = pyquil.get_qc('9q-square')
        
    def get_backend(self):
        return self.qvm
        
    def execute_circuit(self, program, shots=1000):
        executable = self.qvm.compile(program)
        return self.qvm.run(executable)
