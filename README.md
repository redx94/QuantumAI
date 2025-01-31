# QuantumAI 🧠⚛️

> Bridging the gap between Quantum Computing and Artificial Intelligence: An open-source framework for quantum-enhanced machine learning algorithms and AI model optimization.

QuantumAI seamlessly integrates quantum computing capabilities with modern AI techniques, offering unprecedented computational advantages for machine learning tasks. Perfect for researchers, developers, and organizations looking to explore the frontier of quantum-powered artificial intelligence.

## Key Features
- 🚀 Quantum-enhanced neural networks
- 🔄 Hybrid classical-quantum optimization
- 🎯 Quantum feature mapping
- 🛠️ Easy integration with major quantum hardware providers
- 📊 Built-in quantum ML benchmarking tools

## Prerequisites

- Python 3.9+
- Poetry
- CUDA-capable GPU (optional, for accelerated training)
- System dependencies:
  - gcc/g++
  - python3-dev
  - Build essentials

Important version constraints:
- numpy ==1.23.5
- pennylane ==0.31.0

## Installation

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-dev build-essential gcc g++

# Install dependencies with poetry
poetry config virtualenvs.in-project true
poetry install --no-cache

# If you encounter numpy build issues, try:
poetry run pip install --no-cache-dir numpy==1.23.5
poetry install
```

## Usage

Start the API:
```bash
poetry run uvicorn quantum_ai.api.main:app --reload
```

## Testing

Run the test suite:
```bash
poetry run pytest
```

## Architecture

- `quantum_ai/circuits/`: Quantum circuit implementations
  - Gate-based quantum circuits
  - Variational quantum algorithms
- `quantum_ai/api/`: FastAPI endpoints
  - REST API for quantum computations
  - WebSocket support for real-time results
- `quantum_ai/embeddings/`: Quantum embedding modules
  - Quantum feature maps
  - Hybrid classical-quantum embeddings

## Contributing

1. Fork the repository
2. Create your feature branch
3. Run tests and ensure they pass
4. Submit a pull request

## Documentation

- API docs: `http://localhost:8000/docs`
- [Architecture Overview](docs/architecture.md)
- [Development Guide](docs/development.md)

## License

See [LICENSE.md](LICENSE.md) for the full QuantumAI Open License (QAOL v1.0) terms.