import os
import sys
import json
from anthropic import Anthropic
from openai import OpenAI
import streamlit as st

class CodeGenerator:
    """
    Class for generating code based on prompts and context using 
    AI models like OpenAI GPT-4 and Anthropic Claude.
    """
    
    def __init__(self):
        """Initialize the CodeGenerator class."""
        # Check if API keys are available
        self.openai_available = os.environ.get('OPENAI_API_KEY') is not None
        self.anthropic_available = os.environ.get('ANTHROPIC_API_KEY') is not None
        
        # Initialize clients if API keys are available
        if self.openai_available:
            self.openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        
        if self.anthropic_available:
            self.anthropic_client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
        
        # Define available models
        self.available_models = {
            "openai": ["gpt-4o"] if self.openai_available else [],
            "anthropic": ["claude-3-5-sonnet-20241022"] if self.anthropic_available else []
        }
    
    def check_api_keys(self):
        """
        Check if API keys are available.
        
        Returns:
            dict: Status of API keys
        """
        return {
            "openai_available": self.openai_available,
            "anthropic_available": self.anthropic_available,
            "any_available": self.openai_available or self.anthropic_available
        }
    
    def generate_code(self, prompt, context=None, model_provider="openai", model_name=None, temperature=0.2):
        """
        Generate code based on the prompt and context using the specified model.
        
        Args:
            prompt (str): The main instruction for code generation
            context (str, optional): Additional context or code examples to inform generation
            model_provider (str): The provider of the AI model ("openai" or "anthropic")
            model_name (str, optional): Specific model name to use
            temperature (float): Creativity level (lower is more deterministic)
            
        Returns:
            dict: Generated code and metadata
        """
        # Check if API keys are available
        if model_provider == "openai" and not self.openai_available:
            return {"error": "OpenAI API key not available. Please add it to your environment variables."}
        
        if model_provider == "anthropic" and not self.anthropic_available:
            return {"error": "Anthropic API key not available. Please add it to your environment variables."}
        
        # Determine which model to use
        if model_provider == "openai":
            # Use default model if none specified or specified one not available
            if model_name not in self.available_models["openai"]:
                model_name = "gpt-4o"  # Default to newest model
            
            return self._generate_with_openai(prompt, context, model_name, temperature)
            
        elif model_provider == "anthropic":
            # Use default model if none specified or specified one not available
            if model_name not in self.available_models["anthropic"]:
                model_name = "claude-3-5-sonnet-20241022"  # Default to newest model
                #the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            
            return self._generate_with_anthropic(prompt, context, model_name, temperature)
        
        else:
            return {"error": f"Unsupported model provider: {model_provider}"}
    
    def _generate_with_openai(self, prompt, context, model_name, temperature):
        """
        Generate code using OpenAI's API.
        
        Args:
            prompt (str): The main instruction for code generation
            context (str): Additional context or code examples
            model_name (str): The specific OpenAI model to use
            temperature (float): Creativity level
            
        Returns:
            dict: Generated code and metadata
        """
        # Construct the full prompt with context
        system_message = (
            "You are an expert programmer that generates high-quality, efficient code. "
            "Respond only with working, properly commented code that follows best practices. "
            "Include explanations of how the code works using code comments. "
            "Do not include any other text outside the code response."
        )
        
        full_prompt = prompt
        if context:
            full_prompt = f"Context/examples:\n\n{context}\n\nTask:\n{prompt}"
        
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        try:
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=temperature,
                max_tokens=4000
            )
            
            generated_code = response.choices[0].message.content
            
            return {
                "generated_code": generated_code,
                "model_used": model_name,
                "provider": "openai",
                "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            return {"error": f"Error while generating code with OpenAI: {str(e)}"}
    
    def _generate_with_anthropic(self, prompt, context, model_name, temperature):
        """
        Generate code using Anthropic's API.
        
        Args:
            prompt (str): The main instruction for code generation
            context (str): Additional context or code examples
            model_name (str): The specific Anthropic model to use
            temperature (float): Creativity level
            
        Returns:
            dict: Generated code and metadata
        """
        # Construct the full prompt with context
        system_message = (
            "You are an expert programmer that generates high-quality, efficient code. "
            "Respond only with working, properly commented code that follows best practices. "
            "Include explanations of how the code works using code comments. "
            "Do not include any other text outside the code response."
        )
        
        full_prompt = prompt
        if context:
            full_prompt = f"Context/examples:\n\n{context}\n\nTask:\n{prompt}"
        
        try:
            response = self.anthropic_client.messages.create(
                model=model_name,
                system=system_message,
                max_tokens=4000,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            generated_code = response.content[0].text
            
            return {
                "generated_code": generated_code,
                "model_used": model_name,
                "provider": "anthropic",
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            return {"error": f"Error while generating code with Anthropic: {str(e)}"}
    
    def explain_code(self, code, model_provider="openai", model_name=None):
        """
        Explain the provided code, line by line.
        
        Args:
            code (str): The code to explain
            model_provider (str): The provider of the AI model
            model_name (str, optional): Specific model name to use
            
        Returns:
            dict: Explanation of the code
        """
        prompt = f"Explain the following code line by line, focusing on its purpose and how it works:\n\n```\n{code}\n```"
        
        return self.generate_code(prompt, None, model_provider, model_name, 0.3)
    
    def improve_code(self, code, focus_areas=None, model_provider="openai", model_name=None):
        """
        Suggest improvements for the provided code.
        
        Args:
            code (str): The code to improve
            focus_areas (list, optional): Areas to focus on for improvement
            model_provider (str): The provider of the AI model
            model_name (str, optional): Specific model name to use
            
        Returns:
            dict: Improved version of the code with explanations
        """
        focus_prompt = ""
        if focus_areas and len(focus_areas) > 0:
            focus_prompt = f"Focus on improving these aspects: {', '.join(focus_areas)}."
            
        prompt = (
            f"Improve the following code. {focus_prompt} "
            f"Return the improved version with explanation comments.\n\n```\n{code}\n```"
        )
        
        return self.generate_code(prompt, None, model_provider, model_name, 0.3)
    
    def convert_code(self, code, source_lang, target_lang, model_provider="openai", model_name=None):
        """
        Convert code from one programming language to another.
        
        Args:
            code (str): The code to convert
            source_lang (str): The source programming language
            target_lang (str): The target programming language
            model_provider (str): The provider of the AI model
            model_name (str, optional): Specific model name to use
            
        Returns:
            dict: Converted code with explanations
        """
        prompt = (
            f"Convert the following {source_lang} code to {target_lang}. "
            f"Include comments explaining the conversion:\n\n```{source_lang}\n{code}\n```"
        )
        
        return self.generate_code(prompt, None, model_provider, model_name, 0.3)
    
    def generate_unit_tests(self, code, language, testing_framework=None, model_provider="openai", model_name=None):
        """
        Generate unit tests for the provided code.
        
        Args:
            code (str): The code to test
            language (str): The programming language
            testing_framework (str, optional): Specific testing framework to use
            model_provider (str): The provider of the AI model
            model_name (str, optional): Specific model name to use
            
        Returns:
            dict: Generated unit tests
        """
        framework_prompt = f" using the {testing_framework} framework" if testing_framework else ""
        
        prompt = (
            f"Generate comprehensive unit tests{framework_prompt} for the following {language} code. "
            f"Include edge cases and explanations:\n\n```{language}\n{code}\n```"
        )
        
        return self.generate_code(prompt, None, model_provider, model_name, 0.3)