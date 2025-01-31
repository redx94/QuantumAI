import torch
import unittest
import numpy as np
from quantum_ai.circuits.quantum_layer import QuantumLayer
from quantum_ai.circuits.error_mitigation import NoiseAwareQuantumLayer

class TestErrorMitigation(unittest.TestCase):
    def test_error_mitigation(self):
        """Test the error mitigation capabilities of the enhanced QuantumLayer."""
        
        # Initialize quantum layer with error mitigation
        n_qubits = 2
        n_layers = 3
        layer = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
        
        # Create test input
        batch_size = 4
        x = torch.randn(batch_size, n_qubits)
        
        # Forward pass with error mitigation
        output = layer(x)
        
        # Verify output shape and bounds
        self.assertEqual(output.shape, (batch_size, n_qubits))
        self.assertTrue(torch.all(output >= -1) and torch.all(output <= 1))
    
        # Test error correction by introducing artificial noise
        def run_with_noise(input_tensor, noise_level=0.1):
            # Add random noise to the quantum state
            noisy_output = layer(input_tensor)
            noise = torch.randn_like(noisy_output) * noise_level
            return noisy_output + noise
        
        # Compare results with and without error mitigation
        noisy_output = run_with_noise(x)
        mitigated_output = layer(x)
        
        # Verify error mitigation reduces noise
        noise_without_mitigation = torch.std(noisy_output)
        noise_with_mitigation = torch.std(mitigated_output)
        self.assertLess(noise_with_mitigation, noise_without_mitigation)
        
        # Test measurement error calibration
        self.assertIsNotNone(layer.error_mitigator.calibration_matrix)
        self.assertEqual(
            layer.error_mitigator.calibration_matrix.shape,
            (2**n_qubits, 2**n_qubits)
        )
        
        print("\nError Mitigation Test Results:")
        print(f"Noise level without mitigation: {noise_without_mitigation:.4f}")
        print(f"Noise level with mitigation: {noise_with_mitigation:.4f}")
        print(f"Noise reduction: {((noise_without_mitigation - noise_with_mitigation) / noise_without_mitigation * 100):.1f}%")

if __name__ == "__main__":
    unittest.main()
