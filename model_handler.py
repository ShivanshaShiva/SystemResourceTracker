
"""Handler for different AI models and quantum backends"""
import os
from typing import Optional, Dict, Any
import google.generativeai as genai
import openai
from qiskit_ibm_runtime import QiskitRuntimeService
from config import ModelType, DEFAULT_CONFIGS

class ModelHandler:
    def __init__(self):
        self.configs = DEFAULT_CONFIGS.copy()
        self.initialized_models = {}
        
    def initialize_model(self, model_type: ModelType, api_key: str, **kwargs):
        """Initialize a specific model with API key"""
        if model_type == ModelType.GEMMA:
            self._initialize_gemma(api_key, **kwargs)
        elif model_type == ModelType.CHATGPT:
            self._initialize_chatgpt(api_key)
        elif model_type == ModelType.QISKIT:
            self._initialize_qiskit(api_key)
            
    def _initialize_gemma(self, api_key: str, local_path: Optional[str] = None):
        """Initialize Gemma model - supports both API and local loading"""
        if local_path:
            # Local model loading logic would go here
            # This requires implementing the model loading from mobile storage
            self.configs["gemma"]["local_model_path"] = local_path
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.configs["gemma"]["model_name"])
        self.initialized_models["gemma"] = model
        
    def _initialize_chatgpt(self, api_key: str):
        """Initialize ChatGPT API"""
        openai.api_key = api_key
        self.initialized_models["chatgpt"] = True
        
    def _initialize_qiskit(self, api_key: str):
        """Initialize Qiskit Runtime service"""
        service = QiskitRuntimeService(channel="ibm_quantum", token=api_key)
        self.initialized_models["qiskit"] = service
        
    def generate_text(self, model_type: ModelType, prompt: str) -> str:
        """Generate text using specified model"""
        if model_type == ModelType.GEMMA:
            return self._generate_gemma(prompt)
        elif model_type == ModelType.CHATGPT:
            return self._generate_chatgpt(prompt)
        raise ValueError(f"Text generation not supported for {model_type}")
    
    def _generate_gemma(self, prompt: str) -> str:
        model = self.initialized_models.get("gemma")
        if not model:
            raise ValueError("Gemma model not initialized")
        response = model.generate_content(prompt)
        return response.text
    
    def _generate_chatgpt(self, prompt: str) -> str:
        if "chatgpt" not in self.initialized_models:
            raise ValueError("ChatGPT not initialized")
        response = openai.ChatCompletion.create(
            model=self.configs["chatgpt"]["model"],
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    def run_quantum_job(self, circuit, **kwargs):
        """Run quantum job using Qiskit Runtime"""
        service = self.initialized_models.get("qiskit")
        if not service:
            raise ValueError("Qiskit not initialized")
        backend = service.backend(self.configs["qiskit"]["backend"])
        return backend.run(circuit, **kwargs)
