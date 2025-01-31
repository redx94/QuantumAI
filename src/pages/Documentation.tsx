import React from 'react';
import { ArrowLeft, Terminal, Code, Cpu, Network, Lock, Database, Brain, GitBranch, Zap, Binary, Braces, Globe2 } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Documentation() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-indigo-950 to-purple-900 text-white">
      <div className="container mx-auto px-4 py-16">
        {/* Back Navigation */}
        <Link to="/" className="inline-flex items-center text-blue-400 hover:text-blue-300 mb-8">
          <ArrowLeft className="w-5 h-5 mr-2" />
          Back to Home
        </Link>

        <h1 className="text-5xl font-bold mb-12 gradient-text">QIP-X Apex Documentation</h1>

        {/* Quick Links */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          {[
            { icon: Terminal, title: 'Quick Start', href: '#quick-start' },
            { icon: Code, title: 'API Reference', href: '#api-reference' },
            { icon: Network, title: 'Architecture', href: '#architecture' },
          ].map(({ icon: Icon, title, href }) => (
            <a
              key={title}
              href={href}
              className="bg-white/5 p-6 rounded-lg border border-white/10 hover:border-blue-500/50 transition-all"
            >
              <Icon className="w-8 h-8 text-blue-400 mb-4" />
              <h3 className="text-xl font-semibold">{title}</h3>
            </a>
          ))}
        </div>

        {/* Documentation Sections */}
        <div className="space-y-16">
          {/* Quick Start */}
          <section id="quick-start" className="scroll-mt-16">
            <h2 className="text-3xl font-bold mb-6">Quick Start</h2>
            <div className="bg-white/5 rounded-lg p-8">
              <h3 className="text-xl font-semibold mb-4">1. Environment Setup</h3>
              <pre className="bg-black/50 p-4 rounded-lg overflow-x-auto mb-6">
                <code className="text-sm text-gray-300">
                  {`# Create a new environment
conda create -n quantumAGI python=3.10 -y
conda activate quantumAGI

# Essential ML dependencies
pip install torch transformers accelerate bitsandbytes
pip install pennylane pennylane-qiskit qiskit

# For API & deployment
pip install fastapi uvicorn huggingface_hub

# For encryption & IPFS
pip install pycryptodome py-ipfs-http-client`}
                </code>
              </pre>

              <h3 className="text-xl font-semibold mb-4">2. Initialize Quantum Circuit</h3>
              <pre className="bg-black/50 p-4 rounded-lg overflow-x-auto">
                <code className="text-sm text-gray-300">
                  {`import pennylane as qml
from pennylane import numpy as np

# Initialize quantum device
dev = qml.device("default.qubit", wires=5)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    // ...existing code...
    return [qml.expval(qml.PauliZ(i)) for i in range(5)]`}
                </code>
              </pre>
            </div>
          </section>

          {/* API Reference */}
          <section id="api-reference" className="scroll-mt-16">
            <h2 className="text-3xl font-bold mb-6">API Reference</h2>
            <div className="bg-white/5 rounded-lg p-8">
              <div className="space-y-8">
                <div>
                  <h3 className="text-xl font-semibold text-blue-400 mb-4">Quantum Attention Layer</h3>
                  <pre className="bg-black/50 p-4 rounded-lg overflow-x-auto mb-4">
                    <code className="text-sm text-gray-300">
                      {`class QFTAttentionHead(nn.Module):
    def __init__(self, n_qubits=5):
        super().__init__()
        self.n_qubits = n_qubits
        self.theta = nn.Parameter(torch.zeros(n_qubits))
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
    def forward(self, x):
        // ...existing code...
        return self.qnode(x, self.theta)`}
                    </code>
                  </pre>
                  <p className="text-gray-300">QFT-based attention mechanism for enhanced feature extraction.</p>
                </div>

                <div>
                  <h3 className="text-xl font-semibold text-blue-400 mb-4">Model Configuration</h3>
                  <pre className="bg-black/50 p-4 rounded-lg overflow-x-auto mb-4">
                    <code className="text-sm text-gray-300">
                      {`{
  "model_name": "mistralai/Mistral-7B-v0.1",
  "quantum_config": {
    "n_qubits": 5,
    "backend": "ibm_jakarta",
    "optimization": {
      "noise_model": "basic",
      "depth_optimization": true
    }
  }
}`}
                    </code>
                  </pre>
                  <p className="text-gray-300">Configuration for the quantum-enhanced language model.</p>
                </div>

                <div>
                  <h3 className="text-xl font-semibold text-blue-400 mb-4">REST API Endpoints</h3>
                  <div className="space-y-4">
                    <div className="bg-black/50 p-4 rounded-lg">
                      <h4 className="font-semibold mb-2 text-green-400">POST /api/v1/generate</h4>
                      <p className="text-gray-300 mb-2">Generate text using the quantum-enhanced model.</p>
                      <pre className="text-sm text-gray-300">
                        {`{
  "prompt": "string",
  "max_length": 128,
  "temperature": 0.7
}`}
                      </pre>
                    </div>

                    <div className="bg-black/50 p-4 rounded-lg">
                      <h4 className="font-semibold mb-2 text-green-400">GET /api/v1/model/status</h4>
                      <p className="text-gray-300">Get current model and quantum backend status.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Architecture */}
          <section id="architecture" className="scroll-mt-16">
            <h2 className="text-3xl font-bold mb-6">Architecture</h2>
            <div className="bg-white/5 rounded-lg p-8">
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-xl font-semibold mb-4 flex items-center">
                    <Binary className="w-5 h-5 mr-2 text-blue-400" />
                    Quantum Processing
                  </h3>
                  <ul className="space-y-2 text-gray-300">
                    <li className="flex items-start gap-2">
                      <Zap className="w-4 h-4 text-blue-400 mt-1" />
                      QFT-based attention mechanism
                    </li>
                    <li className="flex items-start gap-2">
                      <Zap className="w-4 h-4 text-blue-400 mt-1" />
                      Hybrid classical-quantum architecture
                    </li>
                    <li className="flex items-start gap-2">
                      <Zap className="w-4 h-4 text-blue-400 mt-1" />
                      Adaptive circuit depth optimization
                    </li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-4 flex items-center">
                    <Brain className="w-5 h-5 mr-2 text-purple-400" />
                    Neural Architecture
                  </h3>
                  <ul className="space-y-2 text-gray-300">
                    <li className="flex items-start gap-2">
                      <Braces className="w-4 h-4 text-purple-400 mt-1" />
                      Mistral-7B foundation
                    </li>
                    <li className="flex items-start gap-2">
                      <Braces className="w-4 h-4 text-purple-400 mt-1" />
                      QLoRA 4-bit quantization
                    </li>
                    <li className="flex items-start gap-2">
                      <Braces className="w-4 h-4 text-purple-400 mt-1" />
                      Quantum-enhanced embeddings
                    </li>
                  </ul>
                </div>
              </div>

              <div className="mt-8">
                <h3 className="text-xl font-semibold mb-4 flex items-center">
                  <Globe2 className="w-5 h-5 mr-2 text-green-400" />
                  Deployment Architecture
                </h3>
                <div className="bg-black/50 p-4 rounded-lg">
                  <pre className="text-sm text-gray-300">
                    {`┌─────────────────┐     ┌──────────────┐     ┌────────────────┐
│ Quantum Backend │ ←── │ Core Model   │ ←── │ FastAPI Server │
│ (IBM/Xanadu)    │     │ (Mistral-7B) │     │ (HF Spaces)    │
└─────────────────┘     └──────────────┘     └────────────────┘
                              ↑                       ↑
                              │                       │
                        ┌──────────────┐     ┌────────────────┐
                        │ IPFS Storage │     │ Cloudflare CDN │
                        └──────────────┘     └────────────────┘`}
                  </pre>
                </div>
              </div>
            </div>
          </section>

          {/* Security */}
          <section id="security" className="scroll-mt-16">
            <h2 className="text-3xl font-bold mb-6">Security & Encryption</h2>
            <div className="bg-white/5 rounded-lg p-8">
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="text-xl font-semibold mb-4 flex items-center">
                    <Lock className="w-5 h-5 mr-2 text-green-400" />
                    Post-Quantum Security
                  </h3>
                  <ul className="space-y-2 text-gray-300">
                    <li>• Kyber-512 for asymmetric encryption</li>
                    <li>• AES-256 for symmetric operations</li>
                    <li>• ZK-SNARKs for computation verification</li>
                  </ul>
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-4 flex items-center">
                    <GitBranch className="w-5 h-5 mr-2 text-purple-400" />
                    DAO Governance
                  </h3>
                  <ul className="space-y-2 text-gray-300">
                    <li>• Multi-sig parameter updates</li>
                    <li>• Token-weighted voting</li>
                    <li>• Transparent upgrade proposals</li>
                  </ul>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}