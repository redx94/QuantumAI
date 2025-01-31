
import unittest
import numpy as np
from q_fabric.hybrid_model import HybridQuantumModel

class TestHybridQuantumModel(unittest.TestCase):
    def test_model_output(self):
        model = HybridQuantumModel()
        inputs = np.random.random((1, 8))
        output = model(inputs)
        self.assertEqual(output.shape, (1, 4))

if __name__ == '__main__':
    unittest.main()