from pydantic import BaseSettings, Field
from typing import Optional
import os

class QuantumAIConfig(BaseSettings):
    # API Configuration
    API_VERSION: str = "v1"
    DEBUG_MODE: bool = Field(default=False, env="DEBUG_MODE")
    
    # Quantum Backend Configuration
    QUANTUM_BACKEND: str = Field(default="aer_simulator", env="QUANTUM_BACKEND")
    SHOTS: int = Field(default=1000, env="QUANTUM_SHOTS")
    
    # Security
    SECRET_KEY: str = Field(default_factory=lambda: os.urandom(32).hex())
    API_KEY_HEADER: str = "X-API-Key"
    
    # Performance
    MAX_PARALLEL_CIRCUITS: int = Field(default=10, env="MAX_PARALLEL_CIRCUITS")
    CACHE_ENABLED: bool = Field(default=True, env="CACHE_ENABLED")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # in seconds
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        env_file = ".env"
        case_sensitive = True

config = QuantumAIConfig()
