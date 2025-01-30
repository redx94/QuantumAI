from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from ..quantum.core import QuantumCompute
from ..ai.model import QuantumAI

app = FastAPI(title="Quantum AI API")

# Initialize components
quantum = QuantumCompute()
quantum_ai = QuantumAI()

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 50
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Generate quantum enhanced text",
                    "max_length": 50
                }
            ]
        }
    }

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy"}

@app.get("/quantum/bell-state")
async def get_bell_state():
    """Generate and measure a Bell state"""
    try:
        circuit = quantum.create_bell_state()
        result = quantum.run_circuit(circuit)
        return {"quantum_result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quantum/generate")
async def generate_text(request: GenerateRequest):
    """Generate text using quantum-enhanced AI"""
    try:
        response = quantum_ai.generate(request.prompt, request.max_length)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quantum/embedding")
async def quantum_embedding(input_vector: list):
    """Quantum-enhanced token embeddings"""
    try:
        result = quantum.quantum_embedding(input_vector)
        return {"quantum_embedding": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))