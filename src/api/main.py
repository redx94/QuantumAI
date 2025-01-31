from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from quantum.core import QuantumCompute
from quantum.security import SecureKeyManager

app = FastAPI(title="QuantumAI API")
compute = QuantumCompute()
key_manager = SecureKeyManager()

class CircuitRequest(BaseModel):
    circuit: dict

@app.get("/status")
def get_status():
    return {"status": "running", "version": "0.1.0"}

@app.post("/execute")
async def execute_circuit(request: CircuitRequest):
    try:
        result = compute.execute(request.circuit)
        return {"result": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
