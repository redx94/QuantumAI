from typing import Dict, List, Optional
from enum import Enum

class CognitiveMode(Enum):
    LEARNING = "learning"
    INFERENCE = "inference"
    META_LEARNING = "meta_learning"

class CognitiveModule:
    """Base class for AGI cognitive components."""
    
    def __init__(self, mode: CognitiveMode):
        self.mode = mode
        self.knowledge_base: Dict[str, Any] = {}
        self.meta_parameters: Dict[str, float] = {}
    
    def adapt(self, feedback: Dict[str, float]) -> None:
        """Adapt behavior based on feedback."""
        if self.mode == CognitiveMode.META_LEARNING:
            self._update_meta_parameters(feedback)
    
    def _update_meta_parameters(self, feedback: Dict[str, float]) -> None:
        """Update meta-learning parameters."""
        for key, value in feedback.items():
            if key in self.meta_parameters:
                self.meta_parameters[key] *= value
