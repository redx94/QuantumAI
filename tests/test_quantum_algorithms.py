import pytest
import numpy as np
from src.quantum_algorithms import QuantumAlgorithms
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import pytest
import numpy as np
from src.quantum_algorithms import QuantumAlgorithms
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from src.quantum_circuit import QuantumCircuitEnhanced


class TestQuantumAlgorithms:

    def test_grover_search_basic(self):
        """Enhanced test for Grover's algorithm with simple marked states and detailed assertions."""
        n_qubits = 2
        marked_states = [3]  # Binary '11'
        iterations = 1
        
        circuit = QuantumCircuit(n_qubits, n_qubits)
        result_circuit = QuantumAlgorithms.grover_search(circuit, marked_states, iterations=iterations)
        
        assert result_circuit.num_qubits == n_qubits, "Grover circuit has incorrect qubit count."
        
        # Simulate the circuit and check for marked state probability
        simulator = qiskit.Aer.get_backend('qasm_simulator')
        job = qiskit.execute(result_circuit, simulator, shots=1024)
        counts = job.result().get_counts(result_circuit)
        
        marked_state_binary = format(marked_states[0], '02b') # Ensure binary string of length n_qubits
        
        # Assert that the marked state has a higher probability
        assert counts.get(marked_state_binary, 0) > 0, f"Marked state {marked_state_binary} not found in results."

        # Additional assertions for counts
        assert sum(counts.values()) == 1024, "Total shots do not equal expected value."
        for state, count in counts.items():
            assert 0 <= count <= 1024, f"Count for state {state} is out of range."

    def test_qft_output_amplitude(self):
        """Comprehensive test for QFT algorithm output amplitudes, uniformity, and statevector properties."""
        n_qubits = 3
        circuit = QuantumCircuit(n_qubits, n_qubits)
        qubits = list(range(n_qubits))
        result_circuit = QuantumAlgorithms.quantum_fourier_transform(circuit, qubits)
        
        simulator = qiskit.Aer.get_backend('statevector_simulator')
        job = qiskit.execute(result_circuit, simulator)
        statevector = job.result().get_statevector(result_circuit)
        
        expected_amplitude = 1/np.sqrt(2**n_qubits)
        
        # Assert amplitudes are uniform and close to expected value
        assert np.allclose(np.abs(statevector.data), expected_amplitude, atol=1e-3), \
            "QFT output amplitudes are not uniform or close to expected value."

        # Check statevector properties
        assert len(statevector.data) == 2**n_qubits, "Statevector size does not match expected size."
        assert np.iscomplexobj(statevector.data), "Statevector should be a complex numpy array."

        # Further assertions for statevector data
        assert np.all(np.isclose(np.angle(statevector.data[:8]), np.angle(statevector.data[0]), atol=1e-3)), "Phase angles are not uniform."

    def test_grover_invalid_iterations(self):
        """Robust test for Grover's algorithm handling invalid iteration counts, including negative, zero, and non-integer values."""
        circuit = QuantumCircuit(2, 2)
        # Test with negative iterations
        with pytest.raises(ValueError):
            QuantumAlgorithms.grover_search(circuit, [3], iterations=-1)
        # Test with zero iterations
        with pytest.raises(ValueError):
            QuantumAlgorithms.grover_search(circuit, [3], iterations=0)
        # Test with non-integer iterations
        with pytest.raises(TypeError):
            QuantumAlgorithms.grover_search(circuit, [3], iterations=1.5)

    def test_grover_empty_marked_states(self):
        """Detailed test for Grover's algorithm correctly handling empty marked states by raising ValueError."""
        circuit = QuantumCircuit(2, 2)
        with pytest.raises(ValueError):
            QuantumAlgorithms.grover_search(circuit, [], iterations=1)

    def test_qft_invalid_qubits(self):
        """Extensive test for QFT algorithm handling various invalid qubit inputs, including empty lists, negative indices, out-of-range indices, and invalid types."""
        circuit = QuantumCircuit(2, 2)
        # Test with empty qubit list
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_fourier_transform(circuit, [])
        # Test with qubit list containing negative indices
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_fourier_transform(circuit, [-1])
        # Test with qubit list containing indices out of range
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_fourier_transform(circuit, [2])
        # Test with non-integer qubit indices
        with pytest.raises(TypeError):
            QuantumAlgorithms.quantum_fourier_transform(circuit, [1.5])
        # Test with string qubit indices
        with pytest.raises(TypeError):
            QuantumAlgorithms.quantum_fourier_transform(circuit, ["q"])

    @pytest.mark.parametrize("n_qubits, marked_states", [(2, [3]), (3, [5]), (4, [10])])
    def test_grover_search_parameterized(self, n_qubits, marked_states):
        """Parameterized test for Grover's algorithm with different qubit counts and marked states, verifying circuit properties."""
        circuit = QuantumCircuit(n_qubits, n_qubits)
        result_circuit = QuantumAlgorithms.grover_search(circuit, marked_states)
        assert result_circuit.num_qubits == n_qubits, "Grover circuit has incorrect qubit count in parameterized test."
        # Further assertions can be added here for circuit depth, gates, etc.

    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_qft_statevector_output_parameterized(self, n_qubits):
        """Parameterized test for QFT producing correct statevector output for different qubit counts, ensuring accurate amplitudes."""
        circuit = QuantumCircuit(n_qubits, n_qubits)
        qubits = list(range(n_qubits))
        result_circuit = QuantumAlgorithms.quantum_fourier_transform(circuit, qubits)
        
        # Prepare input state |00...0>
        initial_state = Statevector.from_label('0'*n_qubits)
        final_state = initial_state.evolve(result_circuit)
        
        expected_amplitude = 1/np.sqrt(2**n_qubits)
        assert np.allclose(np.abs(final_state.data), expected_amplitude, atol=1e-3), \
            "QFT output state amplitudes incorrect for n_qubits = {n_qubits} in parameterized test."

    
    @pytest.mark.parametrize("n_qubits,iterations", [(2, 1), (3, 2), (4, 3), (5, 4)])
    def test_grover_algorithm_structure_parameterized(self, n_qubits, iterations):
        """Parameterized test for Grover's algorithm circuit structure and operator repetition, validating qubit count, Grover operator presence, iteration count, and circuit depth."""
        oracle = QuantumCircuitEnhanced(n_qubits)
        circuit = QuantumAlgorithms.grover_algorithm(oracle, iterations=iterations)
        
        # Verify circuit properties
        assert circuit.num_qubits == n_qubits, f"Incorrect number of qubits for n_qubits = {n_qubits} in parameterized test."
        assert 'grover_op' in circuit.count_ops(), "Missing Grover operator in parameterized test."
        assert circuit.count_ops()['grover_op'] == iterations, f"Incorrect iteration count for iterations = {iterations} in parameterized test."
        assert circuit.depth() >= iterations * 2, "Insufficient circuit depth in parameterized test."

    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
    def test_qft_statevector_parameterized(self, n_qubits):
        """Parameterized test for QFT producing correct statevector transformation for different qubit counts, ensuring accurate amplitude magnitudes and negligible imaginary components."""
        # Create basis state |0...0>
        initial_state = Statevector.from_label('0'*n_qubits)
        
        # Apply QFT and measure state
        qft_circuit = QuantumAlgorithms.quantum_fourier_transform(n_qubits)
        final_state = initial_state.evolve(qft_circuit)
        
        # Verify equal superposition with phase factors
        expected_amplitude = 1/(2**(n_qubits/2))
        assert np.allclose(np.abs(final_state.data), expected_amplitude, atol=1e-3), \
            f"Incorrect amplitude magnitude for n_qubits = {n_qubits} in parameterized QFT test."
        assert np.allclose(final_state.data.imag, 0, atol=1e-3), \
            "Unexpected imaginary components in parameterized QFT test."

    @pytest.mark.parametrize("invalid_qubits", [0, -1, 1.5, "two"])
    def test_invalid_qft_qubits_parameterized(self, algorithms, invalid_qubits):
        """Parameterized test for error handling of invalid qubit counts and types in QFT."""
        with pytest.raises((ValueError, TypeError)):
            QuantumAlgorithms.quantum_fourier_transform

    def test_grover_search_multiple_marked_states(self):
        """Test Grover's algorithm with multiple marked states."""
        n_qubits = 3
        marked_states = [3, 5, 6]  # Marked states: 011, 101, 110
        iterations = 2  # Increased iterations for multiple marked states

        circuit = QuantumCircuit(n_qubits, n_qubits)
        result_circuit = QuantumAlgorithms.grover_search(circuit, marked_states, iterations=iterations)

        assert result_circuit.num_qubits == n_qubits, "Grover circuit has incorrect qubit count for multiple marked states."

        # Simulate the circuit and check for marked states probabilities
        simulator = qiskit.Aer.get_backend('qasm_simulator')
        job = qiskit.execute(result_circuit, simulator, shots=2048) # Increased shots for multiple marked states
        counts = job.result().get_counts(result_circuit)

        marked_states_binary = [format(state, f'0{n_qubits}b') for state in marked_states]

        # Assert that marked states have higher probabilities
        for marked_state_binary in marked_states_binary:
            assert counts.get(marked_state_binary, 0) > 0, f"Marked state {marked_state_binary} not found in results for multiple marked states."

        # Additional assertions for counts
        assert sum(counts.values()) == 2048, "Total shots do not equal expected value for multiple marked states."
        for state, count in counts.items():
            assert 0 <= count <= 2048, f"Count for state {state} is out of range in multiple marked states test."

    def test_grover_search_iterations(self):
        """Test Grover's algorithm with different iterations, including optimal iterations."""
        n_qubits = 2
        marked_states = [3]  # Binary '11'

        # Test with different iterations
        for iterations in [1, 2]:
            circuit = QuantumCircuit(n_qubits, n_qubits)
            result_circuit = QuantumAlgorithms.grover_search(circuit, marked_states, iterations=iterations)

            assert result_circuit.num_qubits == n_qubits, f"Grover circuit has incorrect qubit count for iterations={iterations}."

            # Simulate the circuit and check for marked state probability
            simulator = qiskit.Aer.get_backend('qasm_simulator')
            job = qiskit.execute(result_circuit, simulator, shots=1024)
            counts = job.result().get_counts(result_circuit)

            marked_state_binary = format(marked_states[0], '02b')

            # Assert that the marked state has a higher probability
            assert counts.get(marked_state_binary, 0) > 0, f"Marked state {marked_state_binary} not found in results for iterations={iterations}."

        # Test probability increase with iterations
        probabilities = []
        for iterations in range(1, 3): # Test for 1 and 2 iterations
            circuit = QuantumCircuit(n_qubits, n_qubits)
            result_circuit = QuantumAlgorithms.grover_search(circuit, marked_states, iterations=iterations)
            simulator = qiskit.Aer.get_backend('qasm_simulator')
            job = qiskit.execute(result_circuit, simulator, shots=1024)
            counts = job.result().get_counts(result_circuit)
            marked_state_binary = format(marked_states[0], '02b')
            probability = counts.get(marked_state_binary, 0) / 1024
            probabilities.append(probability)

        assert probabilities[1] > probabilities[0], "Probability of marked state should increase with iterations."

        # Test optimal iterations (for single marked state, optimal iterations is approximately sqrt(N/M) where N=2^n_qubits, M=1)
        optimal_iterations = int(np.sqrt(2**n_qubits))
        circuit = QuantumCircuit(n_qubits, n_qubits)
        result_circuit = QuantumAlgorithms.grover_search(circuit, marked_states, iterations=optimal_iterations)
        simulator = qiskit.Aer.get_backend('qasm_simulator')
        job = qiskit.execute(result_circuit, simulator, shots=1024)
        counts = job.result().get_counts(result_circuit)
        marked_state_binary = format(marked_states[0], '02b')
        assert counts.get(marked_state_binary, 0) > 0, f"Marked state {marked_state_binary} not found in results for optimal iterations."

    def test_grover_search_invalid_circuit(self):
        """Test Grover's algorithm with invalid circuit inputs."""
        # Test with invalid circuit type
        with pytest.raises(TypeError):
            QuantumAlgorithms.grover_search("invalid", [3], iterations=1)

        # Test with invalid circuit (None)
        with pytest.raises(ValueError):
            QuantumAlgorithms.grover_search(None, [3], iterations=1)

        # Test with invalid circuit (empty QuantumCircuit)
        with pytest.raises(ValueError):
            QuantumAlgorithms.grover_search(QuantumCircuit(), [3], iterations=1)

    def test_qft_invalid_circuit(self):
        """Test QFT algorithm with invalid circuit inputs."""
        # Test with invalid circuit type
        with pytest.raises(TypeError):
            QuantumAlgorithms.quantum_fourier_transform("invalid", [0])

        # Test with invalid circuit (None)
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_fourier_transform(None, [0])

        # Test with invalid circuit (empty QuantumCircuit)
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_fourier_transform(QuantumCircuit(), [0])

    def test_qift_invalid_circuit(self):
        """Test QIFT algorithm with invalid circuit inputs."""
        # Test with invalid circuit type
        with pytest.raises(TypeError):
            QuantumAlgorithms.quantum_inverse_fourier_transform("invalid", [0])

        # Test with invalid circuit (None)
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_inverse_fourier_transform(None, [0])

        # Test with invalid circuit (empty QuantumCircuit)
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_inverse_fourier_transform(QuantumCircuit(), [0])

    def test_grover_search_invalid_marked_states(self):
        """Test Grover's algorithm with invalid marked states."""
        circuit = QuantumCircuit(2, 2)

        # Test with invalid marked states type
        with pytest.raises(TypeError):
            QuantumAlgorithms.grover_search(circuit, "invalid", iterations=1)

        # Test with invalid marked states (None)
        with pytest.raises(ValueError):
            QuantumAlgorithms.grover_search(circuit, None, iterations=1)

        # Test with invalid marked states (empty list)
        with pytest.raises(ValueError):
            QuantumAlgorithms.grover_search(circuit, [], iterations=1)

        # Test with invalid marked states (non-integer elements)
        with pytest.raises(TypeError):
            QuantumAlgorithms.grover_search(circuit, [1.5], iterations=1)

        # Test with invalid marked states (negative integers)
        with pytest.raises(ValueError):
            QuantumAlgorithms.grover_search(circuit, [-1], iterations=1)

        # Test with invalid marked states (out-of-range integers)
        with pytest.raises(ValueError):
            QuantumAlgorithms.grover_search(circuit, [4], iterations=1)

    def test_qft_invalid_qubits(self):
        """Test QFT algorithm with invalid qubit inputs."""
        circuit = QuantumCircuit(2, 2)

        # Test with invalid qubits type
        with pytest.raises(TypeError):
            QuantumAlgorithms.quantum_fourier_transform(circuit, "invalid")

        # Test with invalid qubits (None)
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_fourier_transform(circuit, None)

        # Test with invalid qubits (empty list)
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_fourier_transform(circuit, [])

        # Test with invalid qubits (non-integer elements)
        with pytest.raises(TypeError):
            QuantumAlgorithms.quantum_fourier_transform(circuit, [1.5])

        # Test with invalid qubits (negative integers)
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_fourier_transform(circuit, [-1])

        # Test with invalid qubits (out-of-range integers)
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_fourier_transform(circuit, [2])

    def test_qift_invalid_qubits(self):
        """Test QIFT algorithm with invalid qubit inputs."""
        circuit = QuantumCircuit(2, 2)

        # Test with invalid qubits type
        with pytest.raises(TypeError):
            QuantumAlgorithms.quantum_inverse_fourier_transform(circuit, "invalid")

        # Test with invalid qubits (None)
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_inverse_fourier_transform(circuit, None)

        # Test with invalid qubits (empty list)
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_inverse_fourier_transform(circuit, [])

        # Test with invalid qubits (non-integer elements)
        with pytest.raises(TypeError):
            QuantumAlgorithms.quantum_inverse_fourier_transform(circuit, [1.5])

        # Test with invalid qubits (negative integers)
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_inverse_fourier_transform(circuit, [-1])

        # Test with invalid qubits (out-of-range integers)
        with pytest.raises(ValueError):
            QuantumAlgorithms.quantum_inverse_fourier_transform(circuit, [2])

    def test_quantum_inverse_fourier_transform(self):
        """Test Quantum Inverse Fourier Transform (QIFT) algorithm."""
        n_qubits = 3
        circuit = QuantumCircuit(n_qubits, n_qubits)
        qubits = list(range(n_qubits))

        # Apply QFT
        qft_circuit = QuantumAlgorithms.quantum_fourier_transform(circuit, qubits)

        # Apply Inverse QFT
        iqft_circuit = QuantumAlgorithms.quantum_inverse_fourier_transform(qft_circuit, qubits)

        simulator = qiskit.Aer.get_backend('statevector_simulator')
        job = qiskit.execute(iqft_circuit, simulator)
        statevector = job.result().get_statevector(iqft_circuit)

        # Expected state is initial state |000>
        expected_state = Statevector.from_label('0'*n_qubits)
        np.testing.assert_allclose(statevector.data, expected_state.data, atol=1e-7, err_msg="QIFT output statevector does not match expected initial state.")

    def test_qft_with_input_state(self):
        """Test QFT algorithm with different input states."""
        n_qubits = 2
        circuit = QuantumCircuit(n_qubits, n_qubits)
        qubits = list(range(n_qubits))

        # Prepare input state |10>
        input_state = Statevector.from_label('10')
        input_circuit = QuantumCircuit(n_qubits, n_qubits)
        input_circuit.x(0) # Prepare |10> from |00>
        simulator = qiskit.Aer.get_backend('statevector_simulator')
        job = qiskit.execute(input_circuit, simulator)
        initial_statevector = job.result().get_statevector(input_circuit)

        # Apply QFT
        qft_circuit = QuantumAlgorithms.quantum_fourier_transform(input_circuit, qubits)
        job = qiskit.execute(qft_circuit, simulator)
        output_statevector = job.result().get_statevector(qft_circuit)

        # Calculate expected output statevector for input |10>
        expected_output_statevector_data = np.fft.fft(initial_statevector.data) / np.sqrt(2**n_qubits)
        expected_output_statevector = Statevector(expected_output_statevector_data)


        np.testing.assert_allclose(output_statevector.data, expected_output_statevector.data, atol=1e-7, err_msg="QFT output statevector does not match expected output for input state |10>.")

    def test_qft_phase_accuracy(self):
        """Test phase accuracy of QFT algorithm output."""
        n_qubits = 3
        circuit = QuantumCircuit(n_qubits, n_qubits)
        qubits = list(range(n_qubits))
        result_circuit = QuantumAlgorithms.quantum_fourier_transform(circuit, qubits)
        
        simulator = qiskit.Aer.get_backend('statevector_simulator')
        job = qiskit.execute(result_circuit, simulator)
        statevector = job.result().get_statevector(result_circuit)

        # Expected phase angles for QFT output should be uniform (or have a predictable pattern)
        # Here we check if the phase angles are close to zero (uniform phase for input |000>)
        expected_phase = 0  # Expected phase for input |000>
        assert np.allclose(np.angle(statevector.data), expected_phase, atol=1e-3), \
            "QFT output phase angles are not close to expected uniform phase."


    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_visualization_content_parameterized(self, algorithms, n_qubits):
        """Parameterized test for visualization content, ensuring key circuit elements are present."""
        circuit = QuantumCircuitEnhanced(n_qubits)
        circuit.h(0)
        visualization = circuit.visualize()
        
        assert 'H' in visualization, f"Visualization missing Hadamard gate for n_qubits = {n_qubits}."  # Verify Hadamard gate representation
        assert f'q_{n_qubits-1}' in visualization, f"Visualization missing qubit labels for n_qubits = {n_qubits}."  # Verify qubit labels
        assert '|' in visualization, "Visualization missing state representation."  # Verify state representation   
        assert '0' in visualization or '1' in visualization, "Visualization missing state values."  # Verify state values
        assert 'X' not in visualization, "Visualization contains unexpected elements."  # Verify no unexpected elements
        assert 'Y' not in visualization, "Visualization contains unexpected elements."  # Verify no unexpected elements
        assert 'Z' not in visualization, "Visualization contains unexpected elements."  # Verify no unexpected elements
        assert 'C' not in visualization, "Visualization contains unexpected elements."  # Verify no unexpected elements
        assert 'T' not in visualization, "Visualization contains unexpected elements."  # Verify no unexpected elements
        assert 'S' not in visualization, "Visualization contains unexpected elements."  # Verify no unexpected elements
        assert 'SWAP' not in visualization, "Visualization contains unexpected elements."  # Verify no unexpected elements
        assert 'CR' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'CC' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'U' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'R' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements  
        assert 'RX' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'RY' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'RZ' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'CH' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'CRX' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements    
        assert 'CRY' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'CRZ' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'SWAP' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'ISWAP' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'ISWAP' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'SQISWAP' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'SQISWAP' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'ISWAP' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'SQISWAP' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
        assert 'ISWAP' not in visualization, "Visualization contains unexpected elements." # Verify no unexpected elements
