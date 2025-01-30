# Quantum AI API

A scalable, decentralized Quantum AI API that integrates classical AI with quantum computing capabilities.

## Features

- Hybrid Quantum-Classical AI (LLM + Quantum Variational Layers)
- FastAPI-powered Quantum API
- Quantum state preparation and measurement
- Text generation with quantum enhancement

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/QuantumAI.git
cd QuantumAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the API

Start the API server:
```bash
python run.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /health` - Health check
- `GET /quantum/bell-state` - Generate and measure a Bell state
- `POST /quantum/generate` - Generate text using quantum-enhanced AI

## Example Usage

```python
import requests

# Generate quantum-enhanced text
response = requests.post(
    "http://localhost:8000/quantum/generate",
    json={"prompt": "Hello quantum world!", "max_length": 50}
)
print(response.json())
```

## License

MIT