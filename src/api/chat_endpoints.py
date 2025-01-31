from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from quantum.text_embedding import QuantumTextEmbedding

router = APIRouter()
quantum_processor = QuantumTextEmbedding()

class ChatMessage(BaseModel):
    message: str

@router.post("/chat")
async def process_chat(chat_message: ChatMessage):
    try:
        response = quantum_processor.process_message(chat_message.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
