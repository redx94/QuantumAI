import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = FastAPI()

MODEL_PATH = "quantum_mistral_ft"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).cuda()
model = PeftModel.from_pretrained(base_model, MODEL_PATH).eval()

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128

@app.post("/infer")
def infer(req: PromptRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to("cuda")
    with torch.cuda.amp.autocast():
        gen_tokens = model.generate(**inputs, max_new_tokens=req.max_new_tokens)
    return {"response": tokenizer.decode(gen_tokens[0], skip_special_tokens=True)}

@app.get("/")
def root():
    return {"message": "Quantum-Enhanced Mistral-7B API is running"}
