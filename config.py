
"""Configuration for API keys and model settings"""
from enum import Enum

class ModelType(Enum):
    GEMMA = "gemma"
    CHATGPT = "chatgpt"
    QISKIT = "qiskit"

# Default configurations 
DEFAULT_CONFIGS = {
    "gemma": {
        "model_name": "gemma-7b-it",
        "local_model_path": "/storage/emulated/0/gemma_models/",
        "temperature": 0.7
    },
    "chatgpt": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7
    },
    "qiskit": {
        "provider": "ibm-quantum",
        "backend": "simulator"
    }
}
