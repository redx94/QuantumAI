"""Security Layer: Quantum-safe cryptography and governance controls."""

from enum import Enum

class SecurityLevel(Enum):
    BASIC = "basic"
    QUANTUM_SAFE = "quantum_safe"
    MAXIMUM = "maximum"

class AccessTier(Enum):
    EDUCATIONAL = "educational"
    ENTERPRISE = "enterprise"
    AGI_RESEARCH = "agi_research"
