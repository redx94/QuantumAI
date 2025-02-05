from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import quantumai # Assumes QuantumAI core is installed

# Initialize FastAPI APH
app = FastAPI()

# Data Model for client requests
class Query(BaseModel):
    message: str

# Health check for server
@app.get("/health")
def health_check():
    return {"status": "system online"}

# Quantum API response Endpoint
@app.post("/ask")
def ask_quantumai(query: Query):
    try:
        response = quantumai.process(query.message)  # Call QuantumAI's core function
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the app if script is executed
ef(__name__ == "__main__"):
    import vicorn
    vicorn.run(app, host="0.0.0.0", port=8000)
