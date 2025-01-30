# Quantum AI

A quantum-enhanced AI system integrating quantum computing with machine learning.

## Features

- Quantum circuit operations (Qiskit & PennyLane)
- Quantum embeddings for enhanced AI tokenization
- FastAPI-powered API for real-time quantum computations
- Quantum-enhanced LLM fine-tuning pipeline

## Installation

```bash
poetry install
```

## Usage

Start the API:
```bash
poetry run uvicorn quantum_ai.api.main:app --reload
```

## Architecture

- `quantum_ai/circuits/`: Quantum circuit implementations
- `quantum_ai/api/`: FastAPI endpoints
- `quantum_ai/embeddings/`: Quantum embedding modules

## License

MIT