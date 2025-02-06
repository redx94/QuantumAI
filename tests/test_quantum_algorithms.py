import pytest
import numpy as np
from src.quantum_algorithms import AdvancedQuantumAlgorithms
from src.quantum_circuit import QuantumCircuitEnhanced
from qiskit.quantum_info import Statevector

class TestQuantumAlgorithms:
    @pytest.fixture
    def algorithms(self):
        return AdvancedQuantumAlgorithms()

    @pytest.mark.parametrize("n_qubits,iterations", [(2, 1), (3, 2), (4, 3)])
    def test_grover_algorithm_structure(self, algorithms, n_qubits, iterations):
        """Test Grover's algorithm circuit structure"""
        oracle = Quantum
import pytest
import numpy as np
from src.quantum_algorithms import AdvancedQuantumAlgorithms
from src.quantum_circuit import QuantumCircuitEnhanced
from qiskit.quantum_info import Statevector

class TestQuantumAlgorithms:
    @pytest.fixture
    def algorithms(self):
        return AdvancedQuantumAlgorithms()

    @pytest.mark.parametrize("n_qubits,iterations", [(2, 1), (3, 2), (4, 3), (5, 4)])
    def test_grover_algorithm_structure(self, algorithms, n_qubits, iterations):
        """Test Grover's algorithm circuit structure and operator repetition"""
        oracle = QuantumCircuitEnhanced(n_qubits)
        circuit = algorithms.grover_algorithm(oracle, iterations=iterations)
        
        # Verify circuit properties
        assert circuit.num_qubits == n_qubits, "Incorrect number of qubits"
        assert 'grover_op' in circuit.count_ops(), "Missing Grover operator"
        assert circuit.count_ops()['grover_op'] == iterations, "Incorrect iteration count"
        assert circuit.depth() >= iterations * 2, "Insufficient circuit depth"

    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
    def test_qft_statevector(self, algorithms, n_qubits):
        """Test QFT produces correct statevector transformation"""
        # Create basis state |0...0>
        initial_state = Statevector.from_label('0'*n_qubits)
        
        # Apply QFT and measure state
        qft_circuit = algorithms.quantum_fourier_transform(n_qubits)
        final_state = initial_state.evolve(qft_circuit)
        
        # Verify equal superposition with phase factors
        expected_amplitude = 1/(2**(n_qubits/2))
        assert np.allclose(np.abs(final_state.data), expected_amplitude, atol=1e-3), \
            "Incorrect amplitude magnitude"
        assert np.allclose(final_state.data.imag, 0, atol=1e-3), \
            "Unexpected imaginary components"

    @pytest.mark.parametrize("invalid_qubits", [0, -1, 1.5, "two"])
    def test_invalid_qft_qubits(self, algorithms, invalid_qubits):
        """Test error handling for invalid qubit counts and types"""
        with pytest.raises((ValueError, TypeError)):
            algorithms.quantum_fourier_transform
import pytest
import numpy as np
from src.quantum_algorithms import AdvancedQuantumAlgorithms
from src.quantum_circuit import QuantumCircuitEnhanced
from qiskit.quantum_info import Statevector

class TestQuantumAlgorithms:
    @pytest.fixture
    def algorithms(self):
        return AdvancedQuantumAlgorithms()

    @pytest.mark.parametrize("n_qubits,iterations", [(2, 1), (3, 2), (4, 3)])
    def test_grover_algorithm_structure(self, algorithms, n_qubits, iterations):
        """Test Grover's algorithm circuit structure"""
        oracle = Quantum
import pytest
import numpy as np
from src.quantum_algorithms import AdvancedQuantumAlgorithms
from src.quantum_circuit import QuantumCircuitEnhanced
from qiskit.quantum_info import Statevector

class TestQuantumAlgorithms:
    @pytest.fixture
    def algorithms(self):
        return AdvancedQuantumAlgorithms()

    @pytest.mark.parametrize("n_qubits,iterations", [(2, 1), (3, 2), (4, 3)])
    def test_grover_algorithm_structure(self, algorithms, n_qubits, iterations):
        """Test Grover's algorithm circuit structure"""
        oracle = QuantumCircuitEnhanced(n_qubits)
        circuit = algorithms.grover_algorithm(oracle, iterations=iterations)
        assert circuit.num_qubits == n_qubits
        # Verify Grover operator repetition
        assert circuit.count_ops()['grover_op'] == iterations

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_qft_statevector(self, algorithms, n_qubits):
        """Test QFT produces correct statevector transformation"""
        # Create computational basis state |0...0>
        initial_state = Statevector.from_label('0'*n_qubits)
        
        # Apply QFT
        qft_circuit = algorithms.quantum_fourier_transform(n_qubits)
        final_state = initial_state.evolve(qft_circuit)
        
        # Verify equal superposition state
        expected_amplitude = 1/(2**(n_qubits/2))
        assert np.allclose(final_state.data.real, expected_amplitude, atol=1e-3)

    def test_invalid_qft_qubits(self, algorithms):
        """Test error handling for invalid qubit counts"""
        with pytest.raises(ValueError):
            algorithms.quantum_fourier_transform(0)

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_visualization_content(self, n_qubits):
        """Test visualization contains key circuit elements"""
        circuit = QuantumCircuitEnhanced(n_qubits)
        circuit.h(0)
        visualization = circuit.visualize()
        
        assert 'H' in visualization  # Verify Hadamard gate representation
        assert f'q_{n_qubits-1}' in visualization  # Verify qubit labels
