import re
import json
import os
from typing import Dict, List, Any, Tuple, Optional, Union

class InstructionLearner:
    """
    A class for learning from natural language instructions in any language.
    This handles more general-purpose instruction learning beyond specific Sanskrit grammar rules.
    """
    
    def __init__(self):
        """Initialize the InstructionLearner."""
        # Store learned instructions and their interpretations
        self.learned_instructions = []
        
        # Dictionary for storing key-value mappings learned from instructions
        self.key_value_mappings = {}
        
        # Dictionary for storing numerical operations/equations
        self.numerical_operations = {}
        
        # Dictionary for translations between languages
        self.translations = {}
        
        # Load previously saved instructions if available
        self._load_saved_instructions()
    
    def _load_saved_instructions(self):
        """Load previously saved instructions from a file."""
        instruction_file = "saved_instructions.json"
        if os.path.exists(instruction_file):
            try:
                with open(instruction_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.learned_instructions = data.get("instructions", [])
                    self.key_value_mappings = data.get("mappings", {})
                    self.numerical_operations = data.get("operations", {})
                    self.translations = data.get("translations", {})
            except Exception as e:
                print(f"Error loading saved instructions: {e}")
    
    def save_instructions(self):
        """Save learned instructions to a file."""
        instruction_file = "saved_instructions.json"
        try:
            with open(instruction_file, "w", encoding="utf-8") as f:
                json.dump({
                    "instructions": self.learned_instructions,
                    "mappings": self.key_value_mappings,
                    "operations": self.numerical_operations,
                    "translations": self.translations
                }, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving instructions: {e}")
            return False
    
    def learn_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Learn from a natural language instruction.
        
        Args:
            instruction (str): The instruction text
            
        Returns:
            dict: Information about what was learned
        """
        # Try to identify the type of instruction
        instruction_type = self._identify_instruction_type(instruction)
        
        if instruction_type == "definition":
            return self._learn_definition(instruction)
        elif instruction_type == "equation":
            return self._learn_equation(instruction)
        elif instruction_type == "translation":
            return self._learn_translation(instruction)
        else:
            # Generic instruction learning
            return self._learn_generic_instruction(instruction)
    
    def _identify_instruction_type(self, instruction: str) -> str:
        """
        Identify the type of instruction from text.
        
        Args:
            instruction (str): The instruction text
            
        Returns:
            str: The identified instruction type
        """
        # Check for definition pattern (X means Y)
        if re.search(r'(\w+)\s+(means|is|=|हैं|है|मतलब)\s+(.+)', instruction, re.IGNORECASE):
            return "definition"
        
        # Check for equation pattern (numbers and operations)
        if re.search(r'\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+', instruction):
            return "equation"
        
        # Check for translation pattern
        if "in english" in instruction.lower() or "in hindi" in instruction.lower() or "translate" in instruction.lower():
            return "translation"
        
        # Default to generic instruction
        return "generic"
    
    def _learn_definition(self, instruction: str) -> Dict[str, Any]:
        """
        Learn a definition or meaning from instruction.
        
        Args:
            instruction (str): The instruction text
            
        Returns:
            dict: Information about what was learned
        """
        # Try different patterns for definitions
        patterns = [
            # English pattern: X means Y
            r'(\w+)\s+means\s+(.+)',
            # English pattern: X is Y
            r'(\w+)\s+is\s+(.+)',
            # Equal sign: X = Y
            r'(\w+)\s*=\s*(.+)',
            # Hindi pattern: X का मतलब Y है
            r'(\w+)\s+का\s+मतलब\s+(.+)\s+है',
            # Hindi pattern: X मतलब Y
            r'(\w+)\s+मतलब\s+(.+)',
            # Hindi pattern: X हैं Y
            r'(\w+)\s+हैं\s+(.+)',
            # Hindi pattern: X है Y
            r'(\w+)\s+है\s+(.+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, instruction, re.IGNORECASE)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                
                # Store the mapping
                self.key_value_mappings[key] = value
                
                # Record the learned instruction
                learned = {
                    "type": "definition",
                    "key": key,
                    "value": value,
                    "original_instruction": instruction
                }
                
                self.learned_instructions.append(learned)
                
                return {
                    "success": True,
                    "learned_type": "definition",
                    "details": learned
                }
        
        # No pattern matched
        return {
            "success": False,
            "error": "Could not identify a definition pattern in the instruction",
            "instruction": instruction
        }
    
    def _learn_equation(self, instruction: str) -> Dict[str, Any]:
        """
        Learn a numerical equation or operation.
        
        Args:
            instruction (str): The instruction text containing an equation
            
        Returns:
            dict: Information about what was learned
        """
        # Match equations like "4 + 3 = 7"
        match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=\s*(\d+)', instruction)
        
        if match:
            num1 = int(match.group(1))
            operator = match.group(2)
            num2 = int(match.group(3))
            result = int(match.group(4))
            
            # Validate the equation
            calculated_result = None
            if operator == '+':
                calculated_result = num1 + num2
            elif operator == '-':
                calculated_result = num1 - num2
            elif operator == '*':
                calculated_result = num1 * num2
            elif operator == '/' and num2 != 0:
                calculated_result = num1 / num2
            
            # Check if the calculated result matches the provided result
            if calculated_result is not None and calculated_result == result:
                equation = f"{num1} {operator} {num2} = {result}"
                
                # Store the equation
                self.numerical_operations[equation] = {
                    "num1": num1,
                    "operator": operator,
                    "num2": num2,
                    "result": result
                }
                
                # Record the learned instruction
                learned = {
                    "type": "equation",
                    "equation": equation,
                    "components": {
                        "num1": num1,
                        "operator": operator,
                        "num2": num2,
                        "result": result
                    },
                    "original_instruction": instruction
                }
                
                self.learned_instructions.append(learned)
                
                return {
                    "success": True,
                    "learned_type": "equation",
                    "details": learned
                }
            else:
                return {
                    "success": False,
                    "error": "The equation is mathematically incorrect",
                    "provided_equation": f"{num1} {operator} {num2} = {result}",
                    "correct_result": calculated_result
                }
        
        # No pattern matched
        return {
            "success": False,
            "error": "Could not identify an equation pattern in the instruction",
            "instruction": instruction
        }
    
    def _learn_translation(self, instruction: str) -> Dict[str, Any]:
        """
        Learn a translation between languages.
        
        Args:
            instruction (str): The instruction text containing translation
            
        Returns:
            dict: Information about what was learned
        """
        # English to other language
        en_pattern = r'(\w+)\s+in\s+(\w+)\s+is\s+(\w+)'
        # Other language to English
        other_pattern = r'(\w+)\s+in\s+English\s+is\s+(\w+)'
        
        match_en = re.search(en_pattern, instruction, re.IGNORECASE)
        match_other = re.search(other_pattern, instruction, re.IGNORECASE)
        
        if match_en:
            word = match_en.group(1).strip().lower()
            target_language = match_en.group(2).strip().lower()
            translation = match_en.group(3).strip()
            
            # Store the translation
            if target_language not in self.translations:
                self.translations[target_language] = {}
            
            self.translations[target_language][word] = translation
            
            # Record the learned instruction
            learned = {
                "type": "translation",
                "word": word,
                "language": target_language,
                "translation": translation,
                "original_instruction": instruction
            }
            
            self.learned_instructions.append(learned)
            
            return {
                "success": True,
                "learned_type": "translation",
                "details": learned
            }
        
        if match_other:
            word = match_other.group(1).strip()
            translation = match_other.group(2).strip().lower()
            
            # Store the translation
            if "english" not in self.translations:
                self.translations["english"] = {}
            
            self.translations["english"][word] = translation
            
            # Record the learned instruction
            learned = {
                "type": "translation",
                "word": word,
                "language": "english",
                "translation": translation,
                "original_instruction": instruction
            }
            
            self.learned_instructions.append(learned)
            
            return {
                "success": True,
                "learned_type": "translation",
                "details": learned
            }
        
        # No pattern matched
        return {
            "success": False,
            "error": "Could not identify a translation pattern in the instruction",
            "instruction": instruction
        }
    
    def _learn_generic_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Learn a generic instruction that doesn't fit other categories.
        
        Args:
            instruction (str): The instruction text
            
        Returns:
            dict: Information about what was learned
        """
        # For generic instructions, simply record them as-is
        learned = {
            "type": "generic",
            "instruction": instruction,
            "original_instruction": instruction
        }
        
        self.learned_instructions.append(learned)
        
        return {
            "success": True,
            "learned_type": "generic",
            "details": learned
        }
    
    def apply_learned_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Apply what we've learned to answer a query.
        
        Args:
            query (str): The query text
            
        Returns:
            dict: The result of applying our learned knowledge
        """
        # Check for definition queries
        for key, value in self.key_value_mappings.items():
            if key.lower() in query.lower():
                return {
                    "success": True,
                    "query_type": "definition",
                    "key": key,
                    "result": value,
                    "explanation": f"According to what I've learned, '{key}' means '{value}'."
                }
        
        # Check for equation application queries
        equation_match = re.search(r'(\d+)\s*([\+\-\*\/])\s*(\d+)', query)
        if equation_match:
            num1 = int(equation_match.group(1))
            operator = equation_match.group(2)
            num2 = int(equation_match.group(3))
            
            result = None
            if operator == '+':
                result = num1 + num2
            elif operator == '-':
                result = num1 - num2
            elif operator == '*':
                result = num1 * num2
            elif operator == '/' and num2 != 0:
                result = num1 / num2
            
            if result is not None:
                return {
                    "success": True,
                    "query_type": "equation",
                    "expression": f"{num1} {operator} {num2}",
                    "result": result,
                    "explanation": f"I calculated {num1} {operator} {num2} = {result}"
                }
        
        # Check for translation queries
        for language, translations in self.translations.items():
            for word, translation in translations.items():
                if word.lower() in query.lower() and language.lower() in query.lower():
                    return {
                        "success": True,
                        "query_type": "translation",
                        "word": word,
                        "language": language,
                        "result": translation,
                        "explanation": f"The word '{word}' in {language} is '{translation}'."
                    }
        
        # No applicable knowledge found
        return {
            "success": False,
            "error": "I don't have enough knowledge to answer this query",
            "query": query
        }
    
    def get_learned_knowledge(self, knowledge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get information about all learned instructions.
        
        Args:
            knowledge_type (str, optional): Filter by type of knowledge
            
        Returns:
            list: List of learned instructions
        """
        if knowledge_type:
            return [item for item in self.learned_instructions if item["type"] == knowledge_type]
        return self.learned_instructions