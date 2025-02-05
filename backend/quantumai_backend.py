
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import quantumai  # Assuming QuantumAI is installed and accessible

app = FastAPI()

class Query(BaseModel):
    message: str

@app.get("/")
def read_root():
    return {"message": "QuantumAI API is running"}

@app.post("/ask")
def ask_quantum_ai(query: Query):
    try:
        response = quantumai.process(query.message)  # Call QuantumAI's core function
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
