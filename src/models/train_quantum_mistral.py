import torch
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from quantum_attention import QFTAttentionHead

def load_qft_mistral(checkpoint="mistralai/Mistral-7B-v0.1", n_qubits=5):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(
        checkpoint, device_map="auto", torch_dtype=torch.float16
    )
    # Attach quantum head
    base_model.qft_head = QFTAttentionHead(n_qubits)

    # QLoRA config
    lora_config = LoraConfig(
        r=4,
        lora_alpha=32,
        target_modules=["qft_head"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    peft_model = get_peft_model(base_model, lora_config)
    return peft_model, tokenizer

def main():
    model, tokenizer = load_qft_mistral()
    texts = ["Hello quantum world!", "Quantum computing meets AI."]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

    with torch.cuda.amp.autocast():
        out = model(**inputs, labels=inputs["input_ids"])
    print(f"Training loss: {out.loss.item()}")

    model.save_pretrained("quantum_mistral_ft")
    tokenizer.save_pretrained("quantum_mistral_ft")

if __name__ == "__main__":
    main()
