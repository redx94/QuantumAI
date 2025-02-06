import unittest
import config
from core.circuits import create_anti_error_circuit
import numpy as np
import pytest

def test_quantum_initialization():
    """Tests quantum initialization and basic config loading."""
    # Test quantum backend configuration
    assert isinstance(config.QUANTUM_BACKEND, str), "QUANTUM_BACKEND should be a string"
    assert config.SHOTS > 0, "SHOTS should be a positive integer"

    # Test API Configuration
    assert isinstance(config.API_VERSION, str), "API_VERSION should be a string"
    assert isinstance(config.DEBUG_MODE, bool), "DEBUG_MODE should be a boolean"

    # Test Security Configuration
    assert isinstance(config.SECRET_KEY, str), "SECRET_KEY should be a string"
    assert isinstance(config.API_KEY_HEADER, str), "API_KEY_HEADER should be a string"

    # Test Performance Configuration
    assert config.MAX_PARALLEL_CIRCUITS > 0, "MAX_PARALLEL_CIRCUITS should be a positive integer"
    assert isinstance(config.CACHE_ENABLED, bool), "CACHE_ENABLED should be a boolean"
    assert config.CACHE_TTL > 0, "CACHE_TTL should be a positive integer"

    # Test Logging Configuration
    assert isinstance(config.LOG_LEVEL, str), "LOG_LEVEL should be a string"
    assert isinstance(config.LOG_FORMAT, str), "LOG_FORMAT should be a string"

def test_integration_flow():
    """Tests the integration of quantum and AI components, focusing on QuantumAttentionModel with different n_qubits."""
    for n_qubits in [2, 4, 6]:  # Test with different number of qubits
        from ai_engine.quantum_attention import QuantumAttentionModel

        # Initialize quantum attention model
        attention_model = QuantumAttentionModel(n_qubits=n_qubits)

        # Test with valid input data
        valid_input_data = list(np.linspace(0.1, 0.4, n_qubits)) # Adjust input data length
        quantum_output = attention_model.forward(valid_input_data)

        # Assertions for valid output
        assert quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for valid input"
        assert isinstance(quantum_output, np.ndarray), f"Integration flow failed for n_qubits={n_qubits}: Output is not a numpy array"
        assert quantum_output.shape == (n_qubits,), f"Integration flow failed for n_qubits={n_qubits}: Output shape is incorrect" # Adjust shape assertion
        assert quantum_output.dtype == np.float64, f"Integration flow failed for n_qubits={n_qubits}: Output data type is incorrect"
        assert np.all(quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values"
        assert np.all(quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1"

        # Test with different valid input data
        another_valid_input_data = list(np.linspace(0.5, 0.8, n_qubits)) # Adjust input data length
        another_quantum_output = attention_model.forward(another_valid_input_data)
        assert another_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for another valid input"

        # Test with invalid input data (wrong input type)
        with pytest.raises(TypeError):
            attention_model.forward("invalid input type")

        # Test with invalid input data (wrong input shape)
        with pytest.raises(ValueError):
            attention_model.forward(list(np.linspace(0.1, 0.3, n_qubits-1))) # Input shape should be (n_qubits,)

        # Test with edge case input data (all zeros)
        zero_input_data = [0.0] * n_qubits # Adjust input data length
        zero_quantum_output = attention_model.forward(zero_input_data)
        assert zero_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for zero input"
        assert np.all(zero_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for zero input"
        assert np.all(zero_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for zero input"
        # Check output distribution for zero input (example: uniform distribution)
        assert np.allclose(zero_quantum_output, np.ones(n_qubits)/n_qubits), f"Integration flow failed for n_qubits={n_qubits}: Output distribution for zero input is not uniform"

        # Test with input data summing to more than 1
        high_sum_input_data = list(np.linspace(0.8, 1.5, n_qubits))
        high_sum_quantum_output = attention_model.forward(high_sum_input_data)
        assert high_sum_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for high sum input"
        assert np.all(high_sum_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for high sum input"
        assert np.all(high_sum_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for high sum input"

        # Test with input data containing negative values
        negative_input_data = list(np.linspace(-0.5, 0.5, n_qubits))
        negative_quantum_output = attention_model.forward(negative_input_data)
        assert negative_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for negative input"
        assert np.all(negative_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output does not contain negative values for negative input"
        assert np.all(negative_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for negative input"


        # Test with edge case input data (all ones)
        ones_input_data = [1.0] * n_qubits # Adjust input data length
        ones_quantum_output = attention_model.forward(ones_input_data)
        assert ones_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for ones input"
        assert np.all(ones_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for ones input"
        assert np.all(ones_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for ones input"
        # Check output distribution for ones input (example: peaked distribution)
        assert np.argmax(ones_quantum_output) == n_qubits-1, f"Integration flow failed for n_qubits={n_qubits}: Output distribution for ones input is not peaked at the end"

        # Test with input data with different distribution (e.g., exponential)
        exponential_input_data = list(np.random.exponential(scale=0.5, size=n_qubits))
        exponential_quantum_output = attention_model.forward(exponential_input_data)
        assert exponential_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for exponential input"
        assert np.all(exponential_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for exponential input"
        assert np.all(exponential_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for exponential input"

        # Test with input data with normal distribution
        normal_input_data = list(np.random.normal(loc=0.5, scale=0.2, size=n_qubits))
        normal_quantum_output = attention_model.forward(normal_input_data)
        assert normal_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for normal input"
        assert np.all(normal_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for normal input"
        assert np.all(normal_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for normal input"

        # Test with input data with uniform distribution
        uniform_input_data = list(np.random.uniform(low=0.2, high=0.8, size=n_qubits))
        uniform_quantum_output = attention_model.forward(uniform_input_data)
        assert uniform_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for uniform input"
        assert np.all(uniform_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for uniform input"
        assert np.all(uniform_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for uniform input"

        # Test with input data with uniform distribution and different parameters
        uniform_input_data_2 = list(np.random.uniform(low=0.5, high=1.0, size=n_qubits))
        uniform_quantum_output_2 = attention_model.forward(uniform_input_data_2)
        assert uniform_quantum_output_2 is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for uniform input with different parameters"
        assert np.all(uniform_quantum_output_2 >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for uniform input with different parameters"
        assert np.all(uniform_quantum_output_2 <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for uniform input with different parameters"

        # Test with input data with exponential distribution and different parameters
        exponential_input_data_2 = list(np.random.exponential(scale=1.0, size=n_qubits))
        exponential_quantum_output_2 = attention_model.forward(exponential_input_data_2)
        assert exponential_quantum_output_2 is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for exponential input with different parameters"
        assert np.all(exponential_quantum_output_2 >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for exponential input with different parameters"
        assert np.all(exponential_quantum_output_2 <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for exponential input with different parameters"

        # Test with input data with normal distribution and different parameters
        normal_input_data_2 = list(np.random.normal(loc=0.8, scale=0.5, size=n_qubits))
        normal_quantum_output_2 = attention_model.forward(normal_input_data_2)
        assert normal_quantum_output_2 is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for normal input with different parameters"
        assert np.all(normal_quantum_output_2 >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for normal input with different parameters"
        assert np.all(normal_quantum_output_2 <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for normal input with different parameters"

def test_create_anti_error_circuit():
    """Tests the creation of the anti-error quantum circuit."""
    circuit = create_anti_error_circuit()
    assert circuit is not None
    # Check circuit properties
    assert circuit.num_qubits == 3
    assert len(circuit.data) == 7
    # Check the gates in the circuit
    assert circuit.data[0][0].name == 'h'
    assert circuit.data[1][0].name == 'cx'
    assert circuit.data[2][0].name == 'cx'
    assert circuit.data[3][0].name == 'h'
    assert circuit.data[4][0].name == 'cx'
    assert circuit.data[5][0].name == 'h'
    assert circuit.data[6][0].name == 'barrier'

def test_quantum_attention_model_predict():
    """Tests the predict method of the QuantumAttentionModel."""
    from ai_engine.quantum_attention import QuantumAttentionModel

    # Initialize model
    attention_model = QuantumAttentionModel(n_qubits=4)

    # Mock input data
    mock_input_data = np.array([[0.1, 0.2, 0.3, 0.4]])

    # Call predict method
    prediction = attention_model.predict(mock_input_data)

    # Assertions for prediction output
    assert prediction is not None, "Predict method failed: Output is None"
    assert isinstance(prediction, np.ndarray), "Predict method failed: Output is not a numpy array"
    assert prediction.shape == (1, 4), "Predict method failed: Output shape is incorrect"
    assert prediction.dtype == np.float64, "Predict method failed: Output data type is incorrect"

    # Test with different input shape
    mock_input_data_2 = np.array([[0.5, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.4]])
    predictions_2 = attention_model.predict(mock_input_data_2)
    assert predictions_2.shape == (2, 4), "Predict method failed: Output shape is incorrect for multiple inputs"

    # Test with invalid input type
    with pytest.raises(TypeError):
        attention_model.predict("invalid input type")

    # Test with invalid input shape
    with pytest.raises(ValueError):
        attention_model.predict(np.array([[0.1, 0.2, 0.3]])) # Input shape should be (1, 4) or (n_samples, 4)

    # Test with empty input data
        with pytest.raises(ValueError):
            attention_model.predict(np.array([[]]))

def test_quantum_attention_model_failure_conditions():
    """Tests the QuantumAttentionModel with various failure conditions and invalid inputs."""
    from ai_engine.quantum_attention import QuantumAttentionModel

    # Initialize model
    attention_model = QuantumAttentionModel(n_qubits=4)

    # Test with NaN input data
    nan_input_data = [float('nan')] * 4
    with pytest.raises(ValueError, match="Input data contains NaN values"):  # Expect ValueError due to NaN values
        attention_model.forward(nan_input_data)

    # Test with infinite input data
    inf_input_data = [float('inf')] * 4
    with pytest.raises(ValueError, match="Input data contains infinite values"):  # Expect ValueError due to infinite values
        attention_model.forward(inf_input_data)

    # Test with very large input data
    large_input_data = [1e10] * 4
    with pytest.raises(ValueError, match="Input data contains values exceeding normalization range"):  # Expect ValueError due to very large values exceeding normalization
        attention_model.forward(large_input_data)

    # Test with very small input data
    small_input_data = [-1e-10] * 4
    with pytest.raises(ValueError, match="Input data contains negative values below normalization range"):  # Expect ValueError due to very small negative values
        attention_model.forward(small_input_data)

def test_quantum_attention_model_input_validation():
    """Tests input validation of the QuantumAttentionModel's forward method."""
    from ai_engine.quantum_attention import QuantumAttentionModel

    # Initialize model
    attention_model = QuantumAttentionModel(n_qubits=4)

    # Test with input data of wrong type (string instead of list/np.ndarray)
    with pytest.raises(TypeError, match="Input data must be a list or numpy array"):
        attention_model.forward("invalid input")

    # Test with input data of wrong shape (too short)
    with pytest.raises(ValueError, match="Input data must have the same length as n_qubits"):
        attention_model.forward([0.1, 0.2, 0.3])

    # Test with input data of wrong shape (too long)
    with pytest.raises(ValueError, match="Input data must have the same length as n_qubits"):
        attention_model.forward([0.1, 0.2, 0.3, 0.4, 0.5])

    # Test with input data containing non-numeric elements
    with pytest.raises(ValueError, match="Input data must contain numeric values"):
        attention_model.forward([0.1, 0.2, 0.3, "a"])
import unittest
import config
from core.circuits import create_anti_error_circuit
import numpy as np
import pytest

def test_quantum_initialization():
    """Tests quantum initialization and basic config loading."""
    # Test quantum backend configuration
    assert isinstance(config.QUANTUM_BACKEND, str), "QUANTUM_BACKEND should be a string"
    assert config.SHOTS > 0, "SHOTS should be a positive integer"

    # Test API Configuration
    assert isinstance(config.API_VERSION, str), "API_VERSION should be a string"
    assert isinstance(config.DEBUG_MODE, bool), "DEBUG_MODE should be a boolean"

    # Test Security Configuration
    assert isinstance(config.SECRET_KEY, str), "SECRET_KEY should be a string"
    assert isinstance(config.API_KEY_HEADER, str), "API_KEY_HEADER should be a string"

    # Test Performance Configuration
    assert config.MAX_PARALLEL_CIRCUITS > 0, "MAX_PARALLEL_CIRCUITS should be a positive integer"
    assert isinstance(config.CACHE_ENABLED, bool), "CACHE_ENABLED should be a boolean"
    assert config.CACHE_TTL > 0, "CACHE_TTL should be a positive integer"

    # Test Logging Configuration
    assert isinstance(config.LOG_LEVEL, str), "LOG_LEVEL should be a string"
    assert isinstance(config.LOG_FORMAT, str), "LOG_FORMAT should be a string"

def test_integration_flow():
    """Tests the integration of quantum and AI components, focusing on QuantumAttentionModel with different n_qubits."""
    for n_qubits in [2, 4, 6]:  # Test with different number of qubits
        from ai_engine.quantum_attention import QuantumAttentionModel

        # Initialize quantum attention model
        attention_model = QuantumAttentionModel(n_qubits=n_qubits)

        # Test with valid input data
        valid_input_data = list(np.linspace(0.1, 0.4, n_qubits)) # Adjust input data length
        quantum_output = attention_model.forward(valid_input_data)

        # Assertions for valid output
        assert quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for valid input"
        assert isinstance(quantum_output, np.ndarray), f"Integration flow failed for n_qubits={n_qubits}: Output is not a numpy array"
        assert quantum_output.shape == (n_qubits,), f"Integration flow failed for n_qubits={n_qubits}: Output shape is incorrect" # Adjust shape assertion
        assert quantum_output.dtype == np.float64, f"Integration flow failed for n_qubits={n_qubits}: Output data type is incorrect"
        assert np.all(quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values"
        assert np.all(quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1"

        # Test with different valid input data
        another_valid_input_data = list(np.linspace(0.5, 0.8, n_qubits)) # Adjust input data length
        another_quantum_output = attention_model.forward(another_valid_input_data)
        assert another_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for another valid input"

        # Test with invalid input data (wrong input type)
        with pytest.raises(TypeError):
            attention_model.forward("invalid input type")

        # Test with invalid input data (wrong input shape)
        with pytest.raises(ValueError):
            attention_model.forward(list(np.linspace(0.1, 0.3, n_qubits-1))) # Input shape should be (n_qubits,)

        # Test with edge case input data (all zeros)
        zero_input_data = [0.0] * n_qubits # Adjust input data length
        zero_quantum_output = attention_model.forward(zero_input_data)
        assert zero_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for zero input"
        assert np.all(zero_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for zero input"
        assert np.all(zero_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for zero input"
        # Check output distribution for zero input (example: uniform distribution)
        assert np.allclose(zero_quantum_output, np.ones(n_qubits)/n_qubits), f"Integration flow failed for n_qubits={n_qubits}: Output distribution for zero input is not uniform"

        # Test with input data summing to more than 1
        high_sum_input_data = list(np.linspace(0.8, 1.5, n_qubits))
        high_sum_quantum_output = attention_model.forward(high_sum_input_data)
        assert high_sum_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for high sum input"
        assert np.all(high_sum_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for high sum input"
        assert np.all(high_sum_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for high sum input"

        # Test with input data containing negative values
        negative_input_data = list(np.linspace(-0.5, 0.5, n_qubits))
        negative_quantum_output = attention_model.forward(negative_input_data)
        assert negative_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for negative input"
        assert np.all(negative_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output does not contain negative values for negative input"
        assert np.all(negative_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for negative input"


        # Test with edge case input data (all ones)
        ones_input_data = [1.0] * n_qubits # Adjust input data length
        ones_quantum_output = attention_model.forward(ones_input_data)
        assert ones_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for ones input"
        assert np.all(ones_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for ones input"
        assert np.all(ones_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for ones input"
        # Check output distribution for ones input (example: peaked distribution)
        assert np.argmax(ones_quantum_output) == n_qubits-1, f"Integration flow failed for n_qubits={n_qubits}: Output distribution for ones input is not peaked at the end"

        # Test with input data with different distribution (e.g., exponential)
        exponential_input_data = list(np.random.exponential(scale=0.5, size=n_qubits))
        exponential_quantum_output = attention_model.forward(exponential_input_data)
        assert exponential_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for exponential input"
        assert np.all(exponential_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for exponential input"
        assert np.all(exponential_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for exponential input"

        # Test with input data with normal distribution
        normal_input_data = list(np.random.normal(loc=0.5, scale=0.2, size=n_qubits))
        normal_quantum_output = attention_model.forward(normal_input_data)
        assert normal_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for normal input"
        assert np.all(normal_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for normal input"
        assert np.all(normal_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for normal input"

        # Test with input data with uniform distribution
        uniform_input_data = list(np.random.uniform(low=0.2, high=0.8, size=n_qubits))
        uniform_quantum_output = attention_model.forward(uniform_input_data)
        assert uniform_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for uniform input"
        assert np.all(uniform_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for uniform input"
        assert np.all(uniform_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for uniform input"

def test_create_anti_error_circuit():
    """Tests the creation of the anti-error quantum circuit."""
    circuit = create_anti_error_circuit()
    assert circuit is not None
    # Check circuit properties
    assert circuit.num_qubits == 3
    assert len(circuit.data) == 7
def test_quantum_attention_model_predict():
    """Tests the predict method of the QuantumAttentionModel."""
    from ai_engine.quantum_attention import QuantumAttentionModel

    # Initialize model
    attention_model = QuantumAttentionModel(n_qubits=4)

    # Mock input data
    mock_input_data = np.array([[0.1, 0.2, 0.3, 0.4]])

    # Call predict method
    prediction = attention_model.predict(mock_input_data)

    # Assertions for prediction output
    assert prediction is not None, "Predict method failed: Output is None"
    assert isinstance(prediction, np.ndarray), "Predict method failed: Output is not a numpy array"
    assert prediction.shape == (1, 4), "Predict method failed: Output shape is incorrect"
    assert prediction.dtype == np.float64, "Predict method failed: Output data type is incorrect"

    # Test with different input shape
    mock_input_data_2 = np.array([[0.5, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.4]])
    predictions_2 = attention_model.predict(mock_input_data_2)
    assert predictions_2.shape == (2, 4), "Predict method failed: Output shape is incorrect for multiple inputs"

    # Test with invalid input type
    with pytest.raises(TypeError):
        attention_model.predict("invalid input type")

    # Test with invalid input shape
    with pytest.raises(ValueError):
        attention_model.predict(np.array([[0.1, 0.2, 0.3]])) # Input shape should be (1, 4) or (n_samples, 4)

    # Test with empty input data
        with pytest.raises(ValueError):
            attention_model.predict(np.array([[]]))

def test_quantum_attention_model_failure_conditions():
    """Tests the QuantumAttentionModel with various failure conditions and invalid inputs."""
    from ai_engine.quantum_attention import QuantumAttentionModel

    # Initialize model
    attention_model = QuantumAttentionModel(n_qubits=4)

    # Test with NaN input data
    nan_input_data = [float('nan')] * 4
    with pytest.raises(ValueError, match="Input data contains NaN values"):  # Expect ValueError due to NaN values
        attention_model.forward(nan_input_data)

    # Test with infinite input data
    inf_input_data = [float('inf')] * 4
    with pytest.raises(ValueError, match="Input data contains infinite values"):  # Expect ValueError due to infinite values
        attention_model.forward(inf_input_data)

    # Test with very large input data
    large_input_data = [1e10] * 4
    with pytest.raises(ValueError, match="Input data contains values exceeding normalization range"):  # Expect ValueError due to very large values exceeding normalization
        attention_model.forward(large_input_data)

    # Test with very small input data
    small_input_data = [-1e-10] * 4
    with pytest.raises(ValueError, match="Input data contains negative values below normalization range"):  # Expect ValueError due to very small negative values
        attention_model.forward(small_input_data)
import unittest
import config
from core.circuits import create_anti_error_circuit
import numpy as np
import pytest

def test_quantum_initialization():
    """Tests quantum initialization and basic config loading."""
    # Test quantum backend configuration
    assert isinstance(config.QUANTUM_BACKEND, str), "QUANTUM_BACKEND should be a string"
    assert config.SHOTS > 0, "SHOTS should be a positive integer"

    # Test API Configuration
    assert isinstance(config.API_VERSION, str), "API_VERSION should be a string"
    assert isinstance(config.DEBUG_MODE, bool), "DEBUG_MODE should be a boolean"

    # Test Security Configuration
    assert isinstance(config.SECRET_KEY, str), "SECRET_KEY should be a string"
    assert isinstance(config.API_KEY_HEADER, str), "API_KEY_HEADER should be a string"

    # Test Performance Configuration
    assert config.MAX_PARALLEL_CIRCUITS > 0, "MAX_PARALLEL_CIRCUITS should be a positive integer"
    assert isinstance(config.CACHE_ENABLED, bool), "CACHE_ENABLED should be a boolean"
    assert config.CACHE_TTL > 0, "CACHE_TTL should be a positive integer"

    # Test Logging Configuration
    assert isinstance(config.LOG_LEVEL, str), "LOG_LEVEL should be a string"
    assert isinstance(config.LOG_FORMAT, str), "LOG_FORMAT should be a string"

def test_integration_flow():
    """Tests the integration of quantum and AI components, focusing on QuantumAttentionModel with different n_qubits."""
    for n_qubits in [2, 4, 6]:  # Test with different number of qubits
        from ai_engine.quantum_attention import QuantumAttentionModel

        # Initialize quantum attention model
        attention_model = QuantumAttentionModel(n_qubits=n_qubits)

        # Test with valid input data
        valid_input_data = list(np.linspace(0.1, 0.4, n_qubits)) # Adjust input data length
        quantum_output = attention_model.forward(valid_input_data)

        # Assertions for valid output
        assert quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for valid input"
        assert isinstance(quantum_output, np.ndarray), f"Integration flow failed for n_qubits={n_qubits}: Output is not a numpy array"
        assert quantum_output.shape == (n_qubits,), f"Integration flow failed for n_qubits={n_qubits}: Output shape is incorrect" # Adjust shape assertion
        assert quantum_output.dtype == np.float64, f"Integration flow failed for n_qubits={n_qubits}: Output data type is incorrect"
        assert np.all(quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values"
        assert np.all(quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1"

        # Test with different valid input data
        another_valid_input_data = list(np.linspace(0.5, 0.8, n_qubits)) # Adjust input data length
        another_quantum_output = attention_model.forward(another_valid_input_data)
        assert another_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for another valid input"

        # Test with invalid input data (wrong input type)
        with pytest.raises(TypeError):
            attention_model.forward("invalid input type")

        # Test with invalid input data (wrong input shape)
        with pytest.raises(ValueError):
            attention_model.forward(list(np.linspace(0.1, 0.3, n_qubits-1))) # Input shape should be (n_qubits,)

        # Test with edge case input data (all zeros)
        zero_input_data = [0.0] * n_qubits # Adjust input data length
        zero_quantum_output = attention_model.forward(zero_input_data)
        assert zero_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for zero input"
        assert np.all(zero_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for zero input"
        assert np.all(zero_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for zero input"
        # Check output distribution for zero input (example: uniform distribution)
        assert np.allclose(zero_quantum_output, np.ones(n_qubits)/n_qubits), f"Integration flow failed for n_qubits={n_qubits}: Output distribution for zero input is not uniform"

        # Test with input data summing to more than 1
        high_sum_input_data = list(np.linspace(0.8, 1.5, n_qubits))
        high_sum_quantum_output = attention_model.forward(high_sum_input_data)
        assert high_sum_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for high sum input"
        assert np.all(high_sum_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for high sum input"
        assert np.all(high_sum_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for high sum input"

        # Test with input data containing negative values
        negative_input_data = list(np.linspace(-0.5, 0.5, n_qubits))
        negative_quantum_output = attention_model.forward(negative_input_data)
        assert negative_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for negative input"
        assert np.all(negative_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output does not contain negative values for negative input"
        assert np.all(negative_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for negative input"


        # Test with edge case input data (all ones)
        ones_input_data = [1.0] * n_qubits # Adjust input data length
        ones_quantum_output = attention_model.forward(ones_input_data)
        assert ones_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for ones input"
        assert np.all(ones_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for ones input"
        assert np.all(ones_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for ones input"
        # Check output distribution for ones input (example: peaked distribution)
        assert np.argmax(ones_quantum_output) == n_qubits-1, f"Integration flow failed for n_qubits={n_qubits}: Output distribution for ones input is not peaked at the end"

        # Test with input data with different distribution (e.g., exponential)
        exponential_input_data = list(np.random.exponential(scale=0.5, size=n_qubits))
        exponential_quantum_output = attention_model.forward(exponential_input_data)
        assert exponential_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for exponential input"
        assert np.all(exponential_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for exponential input"
        assert np.all(exponential_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for exponential input"

        # Test with input data with normal distribution
        normal_input_data = list(np.random.normal(loc=0.5, scale=0.2, size=n_qubits))
        normal_quantum_output = attention_model.forward(normal_input_data)
        assert normal_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for normal input"
        assert np.all(normal_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for normal input"
        assert np.all(normal_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for normal input"

        # Test with input data with uniform distribution
        uniform_input_data = list(np.random.uniform(low=0.2, high=0.8, size=n_qubits))
        uniform_quantum_output = attention_model.forward(uniform_input_data)
        assert uniform_quantum_output is not None, f"Integration flow failed for n_qubits={n_qubits}: Quantum output is None for uniform input"
        assert np.all(uniform_quantum_output >= 0), f"Integration flow failed for n_qubits={n_qubits}: Output contains negative values for uniform input"
        assert np.all(uniform_quantum_output <= 1), f"Integration flow failed for n_qubits={n_qubits}: Output contains values greater than 1 for uniform input"

def test_create_anti_error_circuit():
    """Tests the creation of the anti-error quantum circuit."""
    circuit = create_anti_error_circuit()
    assert circuit is not None
    # Check circuit properties
    assert circuit.num_qubits == 3
    assert len(circuit.data) == 7
def test_quantum_attention_model_predict():
    """Tests the predict method of the QuantumAttentionModel."""
    from ai_engine.quantum_attention import QuantumAttentionModel

    # Initialize model
    attention_model = QuantumAttentionModel(n_qubits=4)

    # Mock input data
    mock_input_data = np.array([[0.1, 0.2, 0.3, 0.4]])

    # Call predict method
    prediction = attention_model.predict(mock_input_data)

    # Assertions for prediction output
    assert prediction is not None, "Predict method failed: Output is None"
    assert isinstance(prediction, np.ndarray), "Predict method failed: Output is not a numpy array"
    assert prediction.shape == (1, 4), "Predict method failed: Output shape is incorrect"
    assert prediction.dtype == np.float64, "Predict method failed: Output data type is incorrect"

    # Test with different input shape
    mock_input_data_2 = np.array([[0.5, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.4]])
    predictions_2 = attention_model.predict(mock_input_data_2)
    assert predictions_2.shape == (2, 4), "Predict method failed: Output shape is incorrect for multiple inputs"

    # Test with invalid input type
    with pytest.raises(TypeError):
        attention_model.predict("invalid input type")

    # Test with invalid input shape
    with pytest.raises(ValueError):
        attention_model.predict(np.array([[0.1, 0.2, 0.3]])) # Input shape should be (1, 4) or (n_samples, 4)

    # Test with empty input data
        with pytest.raises(ValueError):
            attention_model.predict(np.array([[]]))

def test_quantum_attention_model_failure_conditions():
    """Tests the QuantumAttentionModel with various failure conditions and invalid inputs."""
    from ai_engine.quantum_attention import QuantumAttentionModel

    # Initialize model
    attention_model = QuantumAttentionModel(n_qubits=4)

    # Test with NaN input data
    nan_input_data = [float('nan')] * 4
    with pytest.raises(ValueError, match="Input data contains NaN values"):  # Expect ValueError due to NaN values
        attention_model.forward(nan_input_data)

    # Test with infinite input data
    inf_input_data = [float('inf')] * 4
    with pytest.raises(ValueError, match="Input data contains infinite values"):  # Expect ValueError due to infinite values
        attention_model.forward(inf_input_data)

    # Test with very large input data
    large_input_data = [1e10] * 4
    with pytest.raises(ValueError, match="Input data contains values exceeding normalization range"):  # Expect ValueError due to very large values exceeding normalization
        attention_model.forward(large_input_data)

    # Test with very small input data
    small_input_data = [-1e-10] * 4
    with pytest.raises(ValueError, match="Input data contains negative values below normalization range"):  # Expect ValueError due to very small negative values
        attention_model.forward(small_input_data)

    # Test with input data of wrong type (string instead of list/np.ndarray)
    with pytest.raises(TypeError, match="Input data must be a list or numpy array"):
        attention_model.forward("invalid input")

    # Test with input data of wrong shape (too short)
    with pytest.raises(ValueError, match="Input data must have the same length as n_qubits"):
        attention_model.forward([0.1, 0.2, 0.3])

    # Test with input data of wrong shape (too long)
    with pytest.raises(ValueError, match="Input data must have the same length as n_qubits"):
        attention_model.forward([0.1, 0.2, 0.3, 0.4, 0.5])

    # Test with input data containing non-numeric elements
    with pytest.raises(ValueError, match="Input data must contain numeric values"):
        attention_model.forward([0.1, 0.2, 0.3, "a"])
