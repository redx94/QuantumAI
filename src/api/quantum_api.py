from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from typing import List, Optional, Dict, Any
import asyncio
from pydantic import BaseModel
import numpy as np
import redis
from ..quantum.core import QuantumCompute
from ..ai.model import QuantumAI

app = FastAPI(title="Quantum AI API")
api_key_header = APIKeyHeader(name="X-API-Key")
quantum = QuantumCompute()
quantum_ai = QuantumAI()
quantum_compute = QuantumCompute()
quantum_computer = QuantumCompute(n_qubits=8)
redis_client = redis.Redis(host='localhost', port=6379, db=0)

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

class QuantumCircuitRequest(BaseModel):
    circuit_type: str
    parameters: Dict[str, Any]
    n_qubits: Optional[int] = 8

class QuantumJob(BaseModel):
    circuit_type: str
    parameters: Dict[str, Any]
    n_shots: int = 1000

class QuantumResponse(BaseModel):
    job_id: str
    status: str
    result: Dict[str, Any] = None

class QuantumResponse(BaseModel):
    results: List[float]
    metadata: Dict[str, Any]

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != "your-secure-api-key":  # In production, use proper key verification
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

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
        return {"result": result}
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
async def quantum_embedding(input_vector: List[float]):
    """Quantum-enhanced token embeddings"""
    try:
        result = quantum.quantum_embedding(input_vector)
        return {"quantum_embedding": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quantum/status")
async def get_status(api_key: str = Depends(verify_api_key)):
    """Get the current status of the quantum computer"""
    return {
        "status": "operational",
        "available_qubits": quantum_computer.n_qubits,
        "backend": "PennyLane"
    }

@app.post("/quantum/execute", response_model=QuantumResponse)
async def execute_quantum_circuit(
    job: QuantumJob,
    api_key: str = Depends(verify_api_key)
):
    """Execute a quantum circuit asynchronously"""
    job_id = f"job_{np.random.randint(1000000)}"
    
    # Store job in Redis
    redis_client.hset(
        f"quantum_job:{job_id}",
        mapping={
            "status": "pending",
            "circuit_type": job.circuit_type,
            "parameters": str(job.parameters)
        }
    )
    
    # Execute job asynchronously
    asyncio.create_task(process_quantum_job(job_id, job))
    
    return QuantumResponse(
        job_id=job_id,
        status="pending"
    )

@app.post("/quantum/embedding")
async def create_quantum_embedding(
    data: List[List[float]],
    api_key: str = Depends(verify_api_key)
):
    """Create quantum embeddings for input data"""
    try:
        embeddings = quantum_computer.quantum_embedding(np.array(data))
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_quantum_job(job_id: str, job: QuantumJob):
    """Process a quantum job asynchronously"""
    try:
        if job.circuit_type == "vqc":
            result = quantum_computer.vqc_circuit(
                job.parameters["params"],
                job.parameters["input"]
            )
        elif job.circuit_type == "grover":
            result = quantum_computer.execute_grover(
                job.parameters["oracle"],
                job.parameters["iterations"]
            )
        else:
            raise ValueError(f"Unknown circuit type: {job.circuit_type}")
        
        # Update job status in Redis
        redis_client.hset(
            f"quantum_job:{job_id}",
            mapping={
                "status": "completed",
                "result": str(result.tolist())
            }
        )
    except Exception as e:
        redis_client.hset(
            f"quantum_job:{job_id}",
            mapping={
                "status": "failed",
                "error": str(e)
            }
        )