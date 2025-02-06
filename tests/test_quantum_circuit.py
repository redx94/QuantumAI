import pytest
import numpy as np
from src.quantum_circuit import QuantumCircuitEnhanced
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.quantum_info import Statevector
from qiskit.providers.basic_provider import BasicSimulator

class TestQuantumCircuitEnhanced:

    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
    def test_circuit_initialization(self, n_qubits):
        """Test circuit initializes with correct number of qubits."""
        circuit = QuantumCircuitEnhanced(n_qubits)
        assert circuit.num_qubits == n_qubits, "Circuit initialization failed: incorrect qubit count."

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_add_gates(self, n_qubits):
        """Test adding various gates to the quantum circuit."""
        circuit = QuantumCircuitEnhanced(n_qubits)
        circuit.h(0)
        circuit.x(1)
        circuit.cx(0, 1)
        assert circuit.depth() > 0, "Gates not correctly added to the circuit."

        # Test adding more gate combinations and sequences
        circuit = QuantumCircuitEnhanced(n_qubits)
        circuit.h(range(n_qubits)) # Apply H to all qubits
        circuit.rx(np.pi/2, 0) # RX gate
        circuit.ry(np.pi/4, 1) # RY gate
        circuit.rz(np.pi/8, 2) # RZ gate
        circuit.cz(0, 2) # CZ gate
        circuit.swap(1, 2) # SWAP gate
        assert circuit.depth() > 0, "Multiple gate combinations not correctly added to the circuit."

        # Test adding control gates
        circuit = QuantumCircuitEnhanced(n_qubits)
        if n_qubits > 1:
            circuit.ch(0, 1) # CH gate
            circuit.crz(np.pi/3, 0, 1) # CRZ gate
            circuit.cswap(0, 1, 2) # Toffoli/CSWAP gate (requires 3 qubits)
            assert circuit.depth() > 0, "Control gates not correctly added to the circuit."

    def test_add_gates_error_handling(self):
        """Test error handling when adding invalid gates or applying gates to invalid qubits."""
        n_qubits = 2
        circuit = QuantumCircuitEnhanced(n_qubits)

        # Test adding invalid gate type
        with pytest.raises(AttributeError):
            circuit.invalid_gate(0) # type: ignore

        # Test applying gate to invalid qubit index
        with pytest.raises(ValueError):
            circuit.h(n_qubits) # Qubit index out of range

        if n_qubits > 1:
            # Test applying controlled gate with invalid control qubit index
            with pytest.raises(ValueError):
                circuit.cx(n_qubits, 0) # Control qubit index out of range

            # Test applying controlled gate with invalid target qubit index
            with pytest.raises(ValueError):
                circuit.cx(0, n_qubits) # Target qubit index out of range

    @pytest.mark.parametrize("optimization_level", [0, 1, 2, 3])
    def test_transpile_circuit_options(self, optimization_level):
        """Test circuit transpilation with different optimization levels."""
        n_qubits = 2
        circuit = QuantumCircuitEnhanced(n_qubits)
        circuit.h(0)
        circuit.cx(0, 1)

        # Test with different optimization levels
        simulator = BasicSimulator()
        transpiled_circuit = circuit.transpile_circuit(simulator, optimization_level=optimization_level)

        assert isinstance(transpiled_circuit, QuantumCircuit), f"Transpilation with optimization_level={optimization_level} should return a QuantumCircuit object."
        assert transpiled_circuit.depth() <= circuit.depth(), f"Transpiled circuit depth with optimization_level={optimization_level} should not be greater than original."
        # Optionally, add assertions to check if depth decreases with higher optimization levels (though not guaranteed)

        # Test with Aer simulator
        aer_simulator = Aer.get_backend('qasm_simulator')
        aer_transpiled_circuit = circuit.transpile_circuit(aer_simulator, optimization_level=optimization_level)
        assert isinstance(aer_transpiled_circuit, QuantumCircuit), f"Transpilation with Aer simulator and optimization_level={optimization_level} should return a QuantumCircuit object."

    def test_combine_circuits_different_types(self):
        """Test combining QuantumCircuitEnhanced with different circuit types and qubit counts."""
        n_qubits = 2
        enhanced_circuit = QuantumCircuitEnhanced(n_qubits)
        enhanced_circuit.h(0)

        # Test combining with standard QuantumCircuit with different qubit counts (should raise error)
        standard_circuit_diff_qubits = qiskit.QuantumCircuit(n_qubits + 1) # Different qubit count
        with pytest.raises(ValueError):
            enhanced_circuit.combine(standard_circuit_diff_qubits)

        # Test combining with empty QuantumCircuit
        standard_circuit_empty = qiskit.QuantumCircuit(n_qubits) # Same qubit count
        combined_circuit_empty = enhanced_circuit.combine(standard_circuit_empty)
        assert isinstance(combined_circuit_empty, QuantumCircuit), "Combined circuit with empty QuantumCircuit should be a QuantumCircuit object."
        assert combined_circuit_empty.num_qubits == n_qubits, "Combined circuit with empty QuantumCircuit should have the correct number of qubits."
        assert combined_circuit_empty.depth() >= enhanced_circuit.depth(), "Combined circuit with empty QuantumCircuit depth is unexpectedly low."

        # Test combining with another QuantumCircuitEnhanced
        another_enhanced_circuit = QuantumCircuitEnhanced(n_qubits)
        another_enhanced_circuit.x(1)
        combined_enhanced_circuit = enhanced_circuit.combine(another_enhanced_circuit)
        assert isinstance(combined_enhanced_circuit, QuantumCircuit), "Combined circuit with another QuantumCircuitEnhanced should be a QuantumCircuit object."
        assert combined_enhanced_circuit.num_qubits == n_qubits, "Combined circuit with another QuantumCircuitEnhanced should have the correct number of qubits."
        assert combined_enhanced_circuit.depth() >= enhanced_circuit.depth() + another_enhanced_circuit.depth(), "Combined circuit with another QuantumCircuitEnhanced depth is unexpectedly low."


    def test_measure_circuit(self):
        """Test measurement function adds measurement to all qubits."""
import pytest
import numpy as np
from src.quantum_circuit import QuantumCircuitEnhanced
from qiskit import QuantumCircuit, transpile, Aer
from qiskit.quantum_info import Statevector
from qiskit.providers.basic_provider import BasicSimulator

class TestQuantumCircuitEnhanced:

    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
    def test_circuit_initialization(self, n_qubits):
        """Test circuit initializes with correct number of qubits."""
        circuit = QuantumCircuitEnhanced(n_qubits)
        assert circuit.num_qubits == n_qubits, "Circuit initialization failed: incorrect qubit count."

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_add_gates(self, n_qubits):
        """Test adding various gates to the quantum circuit."""
        circuit = QuantumCircuitEnhanced(n_qubits)
        circuit.h(0)
        circuit.x(1)
        circuit.cx(0, 1)
        assert circuit.depth() > 0, "Gates not correctly added to the circuit."

        # Test adding more gate combinations and sequences
        circuit = QuantumCircuitEnhanced(n_qubits)
        circuit.h(range(n_qubits)) # Apply H to all qubits
        circuit.rx(np.pi/2, 0) # RX gate
        circuit.ry(np.pi/4, 1) # RY gate
        circuit.rz(np.pi/8, 2) # RZ gate
        circuit.cz(0, 2) # CZ gate
        circuit.swap(1, 2) # SWAP gate
        assert circuit.depth() > 0, "Multiple gate combinations not correctly added to the circuit."

        # Test adding control gates
        circuit = QuantumCircuitEnhanced(n_qubits)
        if n_qubits > 1:
            circuit.ch(0, 1) # CH gate
            circuit.crz(np.pi/3, 0, 1) # CRZ gate
            circuit.cswap(0, 1, 2) # Toffoli/CSWAP gate (requires 3 qubits)
            assert circuit.depth() > 0, "Control gates not correctly added to the circuit."

    def test_add_gates_error_handling(self):
        """Test error handling when adding invalid gates or applying gates to invalid qubits."""
        n_qubits = 2
        circuit = QuantumCircuitEnhanced(n_qubits)

        # Test adding invalid gate type
        with pytest.raises(AttributeError):
            circuit.invalid_gate(0) # type: ignore

        # Test applying gate to invalid qubit index
        with pytest.raises(ValueError):
            circuit.h(n_qubits) # Qubit index out of range

        if n_qubits > 1:
            # Test applying controlled gate with invalid control qubit index
            with pytest.raises(ValueError):
                circuit.cx(n_qubits, 0) # Control qubit index out of range

            # Test applying controlled gate with invalid target qubit index
            with pytest.raises(ValueError):
                circuit.cx(0, n_qubits) # Target qubit index out of range

    @pytest.mark.parametrize("optimization_level", [0, 1, 2, 3])
    def test_transpile_circuit_options(self, optimization_level):
        """Test circuit transpilation with different optimization levels."""
        n_qubits = 2
        circuit = QuantumCircuitEnhanced(n_qubits)
        circuit.h(0)
        circuit.cx(0, 1)

        # Test with different optimization levels
        simulator = BasicSimulator()
        transpiled_circuit = circuit.transpile_circuit(simulator, optimization_level=optimization_level)

        assert isinstance(transpiled_circuit, QuantumCircuit), f"Transpilation with optimization_level={optimization_level} should return a QuantumCircuit object."
        assert transpiled_circuit.depth() <= circuit.depth(), f"Transpiled circuit depth with optimization_level={optimization_level} should not be greater than original."
        # Optionally, add assertions to check if depth decreases with higher optimization levels (though not guaranteed)

        # Test with Aer simulator
        aer_simulator = Aer.get_backend('qasm_simulator')
        aer_transpiled_circuit = circuit.transpile_circuit(aer_simulator, optimization_level=optimization_level)
        assert isinstance(aer_transpiled_circuit, QuantumCircuit), f"Transpilation with Aer simulator and optimization_level={optimization_level} should return a QuantumCircuit object."


    def test_measure_circuit(self):
        """Test measurement function adds measurement to all qubits."""
        n_qubits = 2
        circuit = QuantumCircuitEnhanced(n_qubits)
        circuit.h(0)
        measured_circuit = circuit.measure_all()
        assert 'measure' in measured_circuit.count_ops(), "Measurement not added to all qubits."
        assert measured_circuit.num_clbits == n_qubits, "Incorrect number of classical bits after measurement."

    def test_measure_circuit_invalid_qubits(self):
        """Test measurement function with invalid qubit counts."""
        # Test with zero qubits
        with pytest.raises(ValueError):
            circuit = QuantumCircuitEnhanced(0)
            circuit.measure_all()

        # Test with negative qubits
        with pytest.raises(ValueError):
            circuit = QuantumCircuitEnhanced(-1)
            circuit.measure_all()

        # Test with non-integer qubits
        with pytest.raises(TypeError):
            circuit = QuantumCircuitEnhanced(2.5)
            circuit.measure_all()

    def test_visualize_circuit_invalid_qubits(self):
        """Test visualization with invalid qubit counts."""
        # Test with zero qubits
        with pytest.raises(ValueError):
            circuit = QuantumCircuitEnhanced(0)
            circuit.visualize()

        # Test with negative qubits
        with pytest.raises(ValueError):
            circuit = QuantumCircuitEnhanced(-1)
            circuit.visualize()

        # Test with non-integer qubits
        with pytest.raises(TypeError):
            circuit = QuantumCircuitEnhanced(2.5)
            circuit.visualize()

    def test_simulate_basic_circuit_invalid_qubits(self):
        """Test simulating a basic quantum circuit with invalid qubit counts."""
        # Test with zero qubits
        with pytest.raises(ValueError):
            circuit = QuantumCircuitEnhanced(0)
            circuit.combine()

        # Test with negative qubits
        with pytest.raises(ValueError):
            circuit = QuantumCircuitEnhanced(-1)
            circuit.combine()

        # Test with non-integer qubits
        with pytest.raises(TypeError):
            circuit = QuantumCircuitEnhanced(2.5)
            circuit.combine()

    def test_transpile_circuit_invalid_qubits(self):
        """Test circuit transpilation with invalid qubit counts."""
        # Test with zero qubits
        with pytest.raises(ValueError):
            circuit = QuantumCircuitEnhanced(0)
            circuit.transpile_circuit(BasicSimulator())

        # Test with negative qubits
        with pytest.raises(ValueError):
            circuit = QuantumCircuitEnhanced(-1)
            circuit.transpile_circuit(BasicSimulator())

        # Test with non-integer qubits
        with pytest.raises(TypeError):
            circuit = QuantumCircuitEnhanced(2.5)
            circuit.transpile_circuit(BasicSimulator())

    def test_combine_circuits_invalid_qubits(self):
        """Test combining circuits with invalid qubit counts."""
        # Test with zero qubits
        with pytest.raises(ValueError):
            circuit = QuantumCircuitEnhanced(0)
            circuit.combine(QuantumCircuit(2))

        # Test with negative qubits
        with pytest.raises(ValueError):
            circuit = QuantumCircuitEnhanced(-1)
            circuit.combine(QuantumCircuit(2))

        # Test with non-integer qubits
        with pytest.raises(TypeError):
            circuit = QuantumCircuitEnhanced(2.5)
            circuit.combine(QuantumCircuit(2))

    def test_get_gates_invalid_qubits(self):
        """Test get_gates method with invalid qubit counts."""
        # Test with zero qubits
        with pytest.raises(ValueError):
            circuit = QuantumCircuitEnhanced(0)
            circuit.get_gates()

        # Test with negative qubits
        with pytest.raises(ValueError):
            circuit = QuantumCircuitEnhanced(-1)
            circuit.get_gates()

        # Test with non-integer qubits
        with pytest.raises(TypeError):
            circuit = QuantumCircuitEnhanced(2.5)
            circuit.get_gates()

    def test_reverse_bits_order_invalid_qubits(self):
        """Test reverse bit order function with invalid qubit counts."""
        # Test with zero qubits
        with pytest.raises(ValueError):
            QuantumCircuitEnhanced.reverse_bits_order({'00': 500, '01': 500}, 0)

        # Test with negative qubits
        with pytest.raises(ValueError):
            QuantumCircuitEnhanced.reverse_bits_order({'00': 500, '01': 500}, -1)

        # Test with non-integer qubits
        with pytest.raises(TypeError):
            QuantumCircuitEnhanced.reverse_bits_order({'00': 500, '01': 500}, 2.5)

    def test_get_histogram_data_invalid_qubits(self):
        """Test get_histogram_data method with invalid qubit counts."""
        # Test with zero qubits
        with pytest.raises(ValueError):
            circuit = QuantumCircuitEnhanced(0)
            circuit.get_histogram_data({'00': 500, '01': 500})

        # Test with negative qubits
        with pytest.raises(ValueError):
            circuit = QuantumCircuitEnhanced(-1)
            circuit.get_histogram_data({'00': 500, '01': 500})

        # Test with non-integer qubits
        with pytest.raises(TypeError):
            circuit = QuantumCircuitEnhanced(2.5)
            circuit.get_histogram_data({'00': 500, '01': 500})

    def test_visualize_circuit(self):
        """Test visualization returns a non-empty string."""
        circuit = QuantumCircuitEnhanced(2)
        visualization = circuit.visualize()
        assert isinstance(visualization, str), "Visualization should return a string."
        assert len(visualization) > 0, "Visualization string is empty."

    def test_error_handling_invalid_qubits(self):
        """Test error handling for invalid qubit counts."""
        with pytest.raises(ValueError):
            QuantumCircuitEnhanced(0)  # Invalid number of qubits
        with pytest.raises(TypeError):
            QuantumCircuitEnhanced("2")  # Invalid qubit type

    def test_simulate_basic_circuit(self):
        """Test simulating a basic quantum circuit and verify the output state."""
        n_qubits = 2
        circuit = QuantumCircuitEnhanced(n_qubits)
        circuit.h(0)  # Apply Hadamard gate to the first qubit
        circuit.cx(0, 1)  # Apply CNOT gate with control qubit 0 and target qubit 1
        
        # Simulate the circuit and get the statevector
        simulator = Aer.get_backend('statevector_simulator')
        quantum_circuit = circuit.combine()
        result = simulator.run(quantum_circuit).result()
        statevector = result.get_statevector(quantum_circuit)

        # Verify the output state
        expected_state = Statevector([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        np.testing.assert_allclose(statevector.data, expected_state.data, atol=1e-7)

    def test_transpile_circuit(self):
        """Test circuit transpilation for optimization."""
        n_qubits = 2
        circuit = QuantumCircuitEnhanced(n_qubits)
        circuit.h(0)
        circuit.cx(0, 1)
        
        # Transpile the circuit
        simulator = BasicSimulator()
        transpiled_circuit = circuit.transpile_circuit(simulator)
        
        assert isinstance(transpiled_circuit, QuantumCircuit), "Transpilation should return a QuantumCircuit object."
        assert transpiled_circuit.depth() <= circuit.depth(), "Transpiled circuit depth should not be greater than original."

    def test_combine_circuits(self):
        """Test combining QuantumCircuitEnhanced with a standard QuantumCircuit."""
        n_qubits = 2
        enhanced_circuit = QuantumCircuitEnhanced(n_qubits)
        enhanced_circuit.h(0)
        
        standard_circuit = QuantumCircuit(n_qubits)
        standard_circuit.cx(0, 1)
        
        combined_circuit = enhanced_circuit.combine(standard_circuit)
        
        assert isinstance(combined_circuit, QuantumCircuit), "Combined circuit should be a QuantumCircuit object."
        assert combined_circuit.num_qubits == n_qubits, "Combined circuit should have the correct number of qubits."
        assert combined_circuit.depth() >= enhanced_circuit.depth() + standard_circuit.depth(), "Combined circuit depth is unexpectedly low."

    def test_get_gates(self):
        """Test get_gates method returns correct gate list."""
        n_qubits = 2
        circuit = QuantumCircuitEnhanced(n_qubits)
        circuit.h(0)
        circuit.cx(0, 1)
        
        gates = circuit.get_gates()
        
        assert isinstance(gates, list), "get_gates should return a list."
        assert all(isinstance(gate, str) for gate in gates), "All elements in gates list should be strings."
        assert 'h' in [gate.lower() for gate in gates], "Hadamard gate should be in the list."
        assert 'cx' in [gate.lower() for gate in gates], "CNOT gate should be in the list."

    def test_reverse_bits_order(self):
        """Test reverse bit order function correctly reverses bit order in counts."""
        counts = {'00': 500, '01': 500}
        reversed_counts = QuantumCircuitEnhanced.reverse_bits_order(counts, 2)
        assert reversed_counts == {'00': 500, '10': 500}, "Bit order reversal failed."

    def test_get_histogram_data(self):
        """Test get_histogram_data method to ensure it returns valid histogram data."""
        circuit = QuantumCircuitEnhanced(2)
        circuit.h(0)
        circuit.measure_all()
        
        simulator = Aer.get_backend('qasm_simulator')
        quantum_circuit = circuit.combine()
        job = simulator.run(quantum_circuit, shots=100)
        counts = job.result().get_counts(quantum_circuit)
        
        histogram_data = circuit.get_histogram_data(counts)
        
        assert isinstance(histogram_data, dict), "Histogram data should be a dictionary."
        assert 'labels' in histogram_data and 'values' in histogram_data, "Histogram data should contain labels and values."
        assert len(histogram_data['labels']) == len(histogram_data['values']), "Labels and values lists should be the same length."
        assert all(isinstance(label, str) for label in histogram_data['labels']), "Histogram labels should be strings."
        assert all(isinstance(value, (int, float)) for value in histogram_data['values']), "Histogram values should be numeric."
