from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .quantum_adapter import QuantumEmbedding
from typing import Optional

class QuantumAI:
    def __init__(self, model_name: str = "gpt2"):
        """Initialize the quantum-enhanced language model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.quantum_adapter = QuantumEmbedding()

    def generate(self, prompt: str, max_length: Optional[int] = 50) -> str:
        """Generate text with quantum enhancement"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        text_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.quantum_adapter.apply_quantum_logic(text_output)
