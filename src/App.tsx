import React from 'react';
import { AlertTriangle, Atom, Brain, CloudLightning, Lock, Scale, Cpu, Database, Network, Zap, GitBranch, Microscope, Calculator, FlaskRound as Flask, Lightbulb, Code2, Boxes, Workflow, Binary, Infinity, Sparkles, Braces, Globe2 } from 'lucide-react';
import { Link } from 'react-router-dom';

function App() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-indigo-950 to-purple-900 text-white">
      <div className="container mx-auto px-4 py-16">
        {/* Hero Section with Quantum Animation */}
        <div className="text-center mb-16 relative">
          <div className="flex justify-center mb-4 relative h-32">
            {/* Central Atom */}
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
              <Atom className="w-16 h-16 text-blue-400 quantum-pulse" />
            </div>
            
            {/* Orbiting Elements */}
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
              <div className="quantum-orbit">
                <Binary className="w-8 h-8 text-green-400 absolute" />
              </div>
              <div className="quantum-orbit-reverse">
                <Brain className="w-8 h-8 text-purple-400 absolute" />
              </div>
              <div className="quantum-orbit-slow">
                <Globe2 className="w-8 h-8 text-pink-400 absolute" />
              </div>
            </div>
          </div>

          <h1 className="text-7xl font-bold mb-6 gradient-text">
            QIP-X Apex
          </h1>
          <p className="text-2xl text-gray-300 max-w-3xl mx-auto">
            The Manhattan Project of AGI: Decentralized, Quantum-Enhanced, Unstoppable
          </p>
        </div>

        {/* Vision Banner */}
        <div className="bg-blue-600/20 border border-blue-500/50 rounded-lg p-8 mb-16">
          <div className="flex items-center gap-4 mb-6">
            <Infinity className="w-10 h-10 text-blue-400" />
            <h2 className="text-2xl font-bold text-blue-200">The Vision</h2>
          </div>
          <p className="text-blue-200 leading-relaxed text-lg">
            Uniting quantum computing, artificial intelligence, and decentralized systems to create
            an unstoppable force in AI evolution. Our hybrid quantum-classical architecture,
            powered by IBM Quantum and Xanadu Borealis, represents a paradigm shift in machine intelligence.
          </p>
        </div>

        {/* Core Technologies Grid */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold mb-8 text-center">Revolutionary Architecture</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="bg-white/10 rounded-lg p-8 backdrop-blur-sm border border-white/10 hover:border-green-500/50 transition-all hover:transform hover:scale-105">
              <Binary className="w-10 h-10 mb-6 text-green-400" />
              <h3 className="text-xl font-bold mb-4">Hybrid Quantum Stack</h3>
              <ul className="space-y-3 text-gray-300">
                <li className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-green-400" />
                  IBM Jakarta QPU Integration
                </li>
                <li className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-green-400" />
                  Xanadu Borealis Photonic QPU
                </li>
                <li className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-green-400" />
                  BitTensor Decentralized Compute
                </li>
              </ul>
            </div>

            <div className="bg-white/10 rounded-lg p-8 backdrop-blur-sm border border-white/10 hover:border-blue-500/50 transition-all hover:transform hover:scale-105">
              <Brain className="w-10 h-10 mb-6 text-blue-400" />
              <h3 className="text-xl font-bold mb-4">QFT-Enhanced Mistral-7B</h3>
              <ul className="space-y-3 text-gray-300">
                <li className="flex items-center gap-2">
                  <Braces className="w-4 h-4 text-blue-400" />
                  Quantum Fourier Transform Attention
                </li>
                <li className="flex items-center gap-2">
                  <Braces className="w-4 h-4 text-blue-400" />
                  4-bit QLoRA Optimization
                </li>
                <li className="flex items-center gap-2">
                  <Braces className="w-4 h-4 text-blue-400" />
                  Self-Tuning Quantum Circuits
                </li>
              </ul>
            </div>

            <div className="bg-white/10 rounded-lg p-8 backdrop-blur-sm border border-white/10 hover:border-purple-500/50 transition-all hover:transform hover:scale-105">
              <Globe2 className="w-10 h-10 mb-6 text-purple-400" />
              <h3 className="text-xl font-bold mb-4">Decentralized Infrastructure</h3>
              <ul className="space-y-3 text-gray-300">
                <li className="flex items-center gap-2">
                  <Network className="w-4 h-4 text-purple-400" />
                  IPFS + Cloudflare Workers
                </li>
                <li className="flex items-center gap-2">
                  <Network className="w-4 h-4 text-purple-400" />
                  Kyber-512 Post-Quantum Security
                </li>
                <li className="flex items-center gap-2">
                  <Network className="w-4 h-4 text-purple-400" />
                  DAO-Driven Evolution
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Technical Innovations */}
        <div className="bg-white/5 rounded-lg p-8 backdrop-blur-sm mb-16 border border-white/10">
          <h2 className="text-3xl font-bold mb-8">Quantum Supremacy Pipeline</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="text-xl font-bold mb-6 text-blue-400">Core Innovations</h3>
              <ul className="space-y-4">
                <li className="flex items-start gap-3">
                  <Zap className="w-6 h-6 text-blue-400 flex-shrink-0 mt-1" />
                  <div>
                    <h4 className="font-semibold mb-1">Quantum Attention Mechanism</h4>
                    <p className="text-gray-400">QFT-based attention heads for superior feature extraction and reasoning</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <Zap className="w-6 h-6 text-blue-400 flex-shrink-0 mt-1" />
                  <div>
                    <h4 className="font-semibold mb-1">Quantum Reinforcement Learning</h4>
                    <p className="text-gray-400">Self-improving circuits via quantum policy optimization</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <Zap className="w-6 h-6 text-blue-400 flex-shrink-0 mt-1" />
                  <div>
                    <h4 className="font-semibold mb-1">Quantum GANs</h4>
                    <p className="text-gray-400">Synthetic dataset generation for continuous model evolution</p>
                  </div>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="text-xl font-bold mb-6 text-purple-400">Tokenomics & Governance</h3>
              <ul className="space-y-4">
                <li className="flex items-start gap-3">
                  <GitBranch className="w-6 h-6 text-purple-400 flex-shrink-0 mt-1" />
                  <div>
                    <h4 className="font-semibold mb-1">$QAI Token</h4>
                    <p className="text-gray-400">Stake for compute priority and revenue sharing</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <GitBranch className="w-6 h-6 text-purple-400 flex-shrink-0 mt-1" />
                  <div>
                    <h4 className="font-semibold mb-1">Quantum Compute NFTs</h4>
                    <p className="text-gray-400">ERC-1155 passes for guaranteed QPU access</p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <GitBranch className="w-6 h-6 text-purple-400 flex-shrink-0 mt-1" />
                  <div>
                    <h4 className="font-semibold mb-1">Community Governance</h4>
                    <p className="text-gray-400">DAO-driven development and parameter updates</p>
                  </div>
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Security Features */}
        <div className="bg-white/5 rounded-lg p-8 backdrop-blur-sm mb-16 border border-white/10">
          <h2 className="text-3xl font-bold mb-8 text-center">Post-Quantum Security</h2>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <Lock className="w-6 h-6 text-green-400 flex-shrink-0 mt-1" />
                <div>
                  <h4 className="font-semibold mb-1">Hybrid Encryption</h4>
                  <p className="text-gray-400">AES-256 + Kyber-512 for quantum-resistant data protection</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-6 h-6 text-yellow-400 flex-shrink-0 mt-1" />
                <div>
                  <h4 className="font-semibold mb-1">Zero-Knowledge Proofs</h4>
                  <p className="text-gray-400">ZK-SNARKs for verifiable computation without data exposure</p>
                </div>
              </div>
            </div>
            <div className="space-y-4">
              <div className="flex items-start gap-3">
                <CloudLightning className="w-6 h-6 text-blue-400 flex-shrink-0 mt-1" />
                <div>
                  <h4 className="font-semibold mb-1">Decentralized Storage</h4>
                  <p className="text-gray-400">IPFS-based model distribution with encrypted sharding</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <Database className="w-6 h-6 text-purple-400 flex-shrink-0 mt-1" />
                <div>
                  <h4 className="font-semibold mb-1">Multi-Sig Governance</h4>
                  <p className="text-gray-400">DAO-controlled parameter updates with threshold signatures</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Call to Action */}
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold mb-6">Join the Quantum Revolution</h2>
          <p className="text-xl text-gray-300 mb-8 max-w-2xl mx-auto">
            Be part of the first decentralized, quantum-enhanced artificial intelligence system.
            Together, we're creating an unstoppable force in AI evolution.
          </p>
          <div className="flex justify-center gap-6">
            <button className="bg-blue-500 hover:bg-blue-600 text-white px-8 py-3 rounded-lg text-lg font-semibold transition-all hover:transform hover:scale-105">
              Join the DAO
            </button>
            <Link 
              to="/docs"
              className="border-2 border-blue-500 hover:bg-blue-500/10 text-white px-8 py-3 rounded-lg text-lg font-semibold transition-all hover:transform hover:scale-105"
            >
              View Documentation
            </Link>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-gray-400">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Lock className="w-5 h-5" />
            <p className="text-lg">Secured with Post-Quantum Cryptography</p>
          </div>
          <p className="text-sm">
            A community-driven initiative combining quantum computing, artificial intelligence, and decentralized governance
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;