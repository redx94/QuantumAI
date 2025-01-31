from qiskit.visualization import plot_circuit_layout, plot_error_map
from qiskit.tools.visualization import plot_histogram
import matplotlib.pyplot as plt
from typing import Optional

class CircuitVisualizer:
    @staticmethod
    def visualize_circuit_layout(circuit, backend) -> None:
        """Visualizes circuit layout on quantum hardware."""
        plot_circuit_layout(circuit, backend)
        plt.show()
    
    @staticmethod
    def analyze_circuit_depth(circuit) -> dict:
        """Analyzes circuit complexity and depth."""
        stats = {
            'depth': circuit.depth(),
            'size': circuit.size(),
            'num_qubits': circuit.num_qubits,
            'ops_breakdown': circuit.count_ops()
        }
        return stats

    @staticmethod
    def plot_results(counts: dict, title: Optional[str] = None) -> None:
        """Plots measurement results."""
        plot_histogram(counts, title=title)
        plt.show()
