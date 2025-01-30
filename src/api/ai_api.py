from fastapi import FastAPI, HTTPException
from ..quantum.embeddings import QuantumEmbedding
from ..training.rl_training import QuantumRL

app = FastAPI(title="Quantum AI Model API")
embedding = QuantumEmbedding()
rl = QuantumRL()

@app.post("/ai/embed")
async def quantum_embed(text: str):
    """Convert text to quantum-enhanced embedding"""
    try:
        # Convert text to numerical representation
        input_vec = [ord(c)/255 for c in text[:4]]
        result = embedding.encode(input_vec)
        return {"embedding": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/predict")
async def quantum_predict(state: list):
    """Get quantum policy prediction"""
    try:
        policy = rl.quantum_policy(state)
        return {"policy": policy.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
