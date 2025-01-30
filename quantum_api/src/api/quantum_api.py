
from fastapi import FastAPI, HTTPException
from ..quantum.core import QuantumCompute
from ..ai.model import QuantumAI

app = FastAPI(title="Quantum API")
quantum = QuantumCompute()
quantum_ai = QuantumAI()

@app.get("/quantum/bell-state")
async def get_bell_state():
    """Generate Bell state & return quantum measurement results"""
    try:
        circuit = quantum.create_bell_state()
        result = quantum.run_circuit(circuit)
        return {"quantum_result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quantum/generate-text")
async def generate_text(prompt: str):
    """Generate text with quantum-enhanced AI"""
    try:
        response = quantum_ai.generate(prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))