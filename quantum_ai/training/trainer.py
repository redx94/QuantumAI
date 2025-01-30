import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..core.circuits import quantum_embedding

class QuantumEnhancedTrainer:
    def __init__(self, model_name: str, n_qubits: int = 4):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.n_qubits = n_qubits

    def quantum_enhanced_forward(self, input_ids, attention_mask=None):
        """Forward pass with quantum-enhanced embeddings."""
        # Convert input_ids to quantum embeddings
        batch_size = input_ids.shape[0]
        quantum_emb = []
        
        for batch in input_ids:
            emb = quantum_embedding(batch.float().numpy(), self.n_qubits)
            quantum_emb.append(torch.from_numpy(emb))
        
        quantum_emb = torch.stack(quantum_emb)
        
        # Combine with traditional embeddings
        outputs = self.model(input_ids, 
                           attention_mask=attention_mask,
                           quantum_embeddings=quantum_emb)
        return outputs

    def train_step(self, batch, optimizer):
        optimizer.zero_grad()
        outputs = self.quantum_enhanced_forward(batch["input_ids"],
                                             batch["attention_mask"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        return loss.item()
