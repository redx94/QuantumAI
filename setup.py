from setuptools import setup, find_packages

setup(
    name="quantum_ai",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "qiskit>=0.39.0",
        "pennylane>=0.28.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-dotenv>=0.19.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
)
