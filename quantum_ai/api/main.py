from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
from ..core.circuits import quantum_embedding, create_variational_circuit
from ..circuits.quantum_layer import QuantumLayer

app = FastAPI(title="QuantumAI API")

class EmbeddingRequest(BaseModel):
    data: list[float]

class CircuitRequest(BaseModel):
    input_data: list[float]
    parameters: list[list[float]]

class QuantumInput(BaseModel):
    data: list[float]
    circuit_type: str

class PredictionRequest(BaseModel):
    inputs: list[float]
    n_qubits: int = 4
    n_layers: int = 2

@app.post("/quantum/embed")
async def create_embedding(request: EmbeddingRequest):
    try:
        embedding = quantum_embedding(np.array(request.data))
        return {"embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quantum/circuit")
async def run_circuit(request: CircuitRequest):
    try:
        circuit = create_variational_circuit(np.array(request.parameters))
        result = circuit(np.array(request.input_data))
        return {"result": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quantum/compute")
async def quantum_compute(input_data: QuantumInput):
    try:
        # Initialize appropriate circuit
        if input_data.circuit_type == "vqc":
            from quantum_ai.circuits.base import VQC
            circuit = VQC(n_qubits=len(input_data.data))
            result = circuit.run(np.array(input_data.data))
            return {"result": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        model = QuantumLayer(request.n_qubits, request.n_layers)
        inputs = torch.tensor(request.inputs).reshape(1, -1)
        result = model(inputs)
        return {"predictions": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
