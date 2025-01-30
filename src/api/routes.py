
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    params: dict = {}

@app.post("/v1/generate")
async def generate(request: GenerationRequest):
    """Generate text using quantum-enhanced LLM"""
    try:
        response = quantum_api.generate(request.prompt)
        return {"generated_text": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/improve")
async def trigger_improvement():
    """Trigger self-improvement cycle"""
    try:
        metrics = self_improver.improve()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))