
import unittest
import numpy as np
from q_fabric.hybrid_model import HybridQuantumModel

class TestHybridQuantumModel(unittest.TestCase):
    def test_model_output(self):
        model = HybridQuantumModel()
        inputs = np.random.random((1, 8))
        output = model(inputs)
        self.assertEqual(output.shape, (1, 4))

    def test_full_pipeline_integration(self):
        # Mock backend and API interactions
        class MockBackend:
            def execute(self, circuit):
                return np.random.random((1, 4))

        class MockAPI:
            def get_data(self, query):
                return np.random.random((1, 8))

        # Initialize components
        model = HybridQuantumModel()
        backend = MockBackend()
        api = MockAPI()

        # Run the pipeline
        data = api.get_data("test_query")
        output = model(data)
        quantum_output = backend.execute(output)

        # Validate the output
        self.assertEqual(quantum_output.shape, (1, 4))

if __name__ == '__main__':
    unittest.main()
