import re
import string
import random

class SanskritNLP:
    """
    Class for handling Sanskrit Natural Language Processing tasks.
    This includes tokenization, grammar analysis, POS tagging, and semantic analysis.
    It can also learn new grammar rules from natural language instructions.
    """
    
    def __init__(self):
        # Load Sanskrit-specific stopwords and other resources
        self.sandhi_rules = self._initialize_sandhi_rules()
        self.pos_tags = self._initialize_pos_tags()
        self.sanskrit_chars = self._initialize_sanskrit_chars()
        
        # Character groups for grammar rules
        self.char_groups = self._initialize_char_groups()
        
        # Store learned grammar rules
        self.learned_rules = []
        self.custom_sandhi_rules = {}
    
    def _initialize_sandhi_rules(self):
        """Initialize basic Sandhi (word joining) rules for Sanskrit."""
        # This is a simplified implementation of basic Sandhi rules
        return {
            "a+a": "ā",
            "a+i": "e",
            "a+u": "o",
            "a+ā": "ā",
            "a+e": "ai",
            "a+o": "au",
            "ā+a": "ā",
            "i+i": "ī",
            "i+a": "ya",
            "u+u": "ū",
            "u+a": "va",
            "t+t": "tt",
            "t+n": "nn",
            "m+t": "nt",
            "m+s": "ms",
            "h+t": "ddh",
            # Additional rules can be added here
        }
    
    def _initialize_pos_tags(self):
        """Initialize Part-of-Speech tags specific to Sanskrit."""
        return {
            "NOUN": "Noun",
            "VERB": "Verb",
            "ADJ": "Adjective",
            "ADV": "Adverb",
            "PRON": "Pronoun",
            "PREP": "Preposition",
            "CONJ": "Conjunction",
            "PART": "Particle",
            "NUM": "Numeral",
            "INDECL": "Indeclinable"
        }
    
    def _initialize_sanskrit_chars(self):
        """Initialize Sanskrit character sets."""
        vowels = ["अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ऌ", "ए", "ऐ", "ओ", "औ"]
        consonants = [
            "क", "ख", "ग", "घ", "ङ", 
            "च", "छ", "ज", "झ", "ञ", 
            "ट", "ठ", "ड", "ढ", "ण", 
            "त", "थ", "द", "ध", "न", 
            "प", "फ", "ब", "भ", "म", 
            "य", "र", "ल", "व", 
            "श", "ष", "स", "ह"
        ]
        marks = ["ा", "ि", "ी", "ु", "ू", "ृ", "ॄ", "ॢ", "े", "ै", "ो", "ौ", "्", "ं", "ः"]
        return {
            "vowels": vowels,
            "consonants": consonants,
            "marks": marks,
            "all": vowels + consonants + marks
        }
        
    def _initialize_char_groups(self):
        """Initialize Sanskrit character groups for grammar rules."""
        # Define common character groups used in Sanskrit grammar
        # These are the groups referred to in rules like "iko yan aci"
        
        # Devanagari notation
        char_groups = {
            # Main groups from the traditional Sanskrit grammar
            "इक्": ["इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ऌ", "ॡ"],  # ik group (i, u, ṛ, ḷ vowels)
            "यण्": ["य", "व", "र", "ल"],  # yaṇ group (semi-vowels)
            "अच्": ["अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ऌ", "ॡ", "ए", "ऐ", "ओ", "औ"],  # ac group (all vowels)
            "हल्": ["क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", 
                    "ण", "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", 
                    "व", "श", "ष", "स", "ह"],  # hal group (all consonants)
            
            # Specific consonant groups
            "कु": ["क", "ख", "ग", "घ", "ङ"],  # ka-varga (gutturals)
            "चु": ["च", "छ", "ज", "झ", "ञ"],  # ca-varga (palatals)
            "टु": ["ट", "ठ", "ड", "ढ", "ण"],  # ṭa-varga (retroflex)
            "तु": ["त", "थ", "द", "ध", "न"],  # ta-varga (dentals)
            "पु": ["प", "फ", "ब", "भ", "म"],  # pa-varga (labials)
            "शल्": ["श", "ष", "स"],  # śal group (sibilants)
            
            # Transliteration notation for ease of use
            "ik": ["i", "ī", "u", "ū", "ṛ", "ṝ", "ḷ", "ḹ"],
            "yan": ["y", "v", "r", "l"],
            "ac": ["a", "ā", "i", "ī", "u", "ū", "ṛ", "ṝ", "ḷ", "ḹ", "e", "ai", "o", "au"],
            "hal": ["k", "kh", "g", "gh", "ṅ", "c", "ch", "j", "jh", "ñ", "ṭ", "ṭh", "ḍ", "ḍh", 
                   "ṇ", "t", "th", "d", "dh", "n", "p", "ph", "b", "bh", "m", "y", "r", "l", 
                   "v", "ś", "ṣ", "s", "h"],
            
            # More specific groups in transliteration
            "ku": ["k", "kh", "g", "gh", "ṅ"],
            "cu": ["c", "ch", "j", "jh", "ñ"],
            "tu": ["ṭ", "ṭh", "ḍ", "ḍh", "ṇ"],
            "tu_": ["t", "th", "d", "dh", "n"],
            "pu": ["p", "ph", "b", "bh", "m"],
            "shal": ["ś", "ṣ", "s"]
        }
        
        return char_groups
    
    def tokenize(self, text):
        """
        Tokenize Sanskrit text into words and sentences.
        
        Args:
            text (str): Sanskrit text to tokenize
            
        Returns:
            dict: Dictionary containing word and sentence tokens
        """
        # Simple tokenization based on spaces and punctuation
        # More sophisticated Sanskrit tokenization would require deeper analysis of sandhi
        
        # Remove any non-Sanskrit characters that might interfere
        # This is a simplified approach - a real implementation would handle transliteration
        # and different text encodings properly
        
        # Split into sentences (approximate using standard punctuation)
        sentences = re.split(r'[।॥.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Split into words
        words = []
        for sentence in sentences:
            # Simple word splitting - in a real implementation, this would 
            # handle Sanskrit-specific word boundaries and sandhi
            sentence_words = re.split(r'[\s,;]+', sentence)
            words.extend([w.strip() for w in sentence_words if w.strip()])
        
        return {
            "words": words,
            "sentences": sentences,
            "word_count": len(words),
            "sentence_count": len(sentences)
        }
    
    def analyze_grammar(self, text):
        """
        Perform grammatical analysis on Sanskrit text.
        
        Args:
            text (str): Sanskrit text to analyze
            
        Returns:
            dict: Dictionary containing grammatical analysis results
        """
        # Tokenize first
        tokens = self.tokenize(text)
        
        # This is a placeholder for more sophisticated grammar analysis
        # A real implementation would analyze Sanskrit grammar based on Paninian rules
        
        # Analyze sandhi (word joining)
        sandhi_instances = []
        for i in range(len(tokens["words"]) - 1):
            current_word = tokens["words"][i]
            next_word = tokens["words"][i + 1]
            
            if current_word and next_word:
                # Check if the ending of the current word + beginning of next word matches any sandhi rule
                if current_word[-1] + "+" + next_word[0] in self.sandhi_rules:
                    sandhi_instances.append({
                        "position": i,
                        "words": [current_word, next_word],
                        "rule": current_word[-1] + "+" + next_word[0],
                        "result": self.sandhi_rules[current_word[-1] + "+" + next_word[0]]
                    })
        
        # Identify potential compounds (समास)
        compounds = []
        long_words = [w for w in tokens["words"] if len(w) > 10]  # Simplistic heuristic
        for word in long_words:
            # In a real implementation, this would involve parsing the compound
            compounds.append({
                "word": word,
                "possible_type": random.choice(["Tatpurusha", "Dvandva", "Bahuvrihi", "Avyayibhava"])
            })
        
        # Identify verb forms (simplified)
        verbs = []
        for word in tokens["words"]:
            # Very simplified verb detection - real implementation would be much more complex
            if word.endswith(("ति", "सि", "मि", "न्ति", "थ", "म:", "ते", "न्ते")):
                verbs.append({
                    "word": word,
                    "possible_form": "Present" if word.endswith(("ति", "सि", "मि")) else "Past" 
                })
        
        return {
            "sandhi_analysis": sandhi_instances,
            "compound_words": compounds,
            "verb_forms": verbs,
            "token_count": tokens["word_count"]
        }
    
    def pos_tag(self, text):
        """
        Perform Part-of-Speech tagging on Sanskrit text.
        
        Args:
            text (str): Sanskrit text to tag
            
        Returns:
            list: List of (token, tag) tuples
        """
        # Tokenize first
        tokens = self.tokenize(text)["words"]
        
        # This is a simplified POS tagging implementation
        # A real Sanskrit POS tagger would use sophisticated rules or a trained model
        
        tagged = []
        for token in tokens:
            # Very simplistic rules - a real implementation would be much more sophisticated
            if token.endswith(("म्", ":", "म")):
                tag = "NOUN"
            elif token.endswith(("ति", "न्ति", "ते", "न्ते", "सि", "थ", "मि", "म:")):
                tag = "VERB"
            elif token.endswith(("तया", "तर", "तम")):
                tag = "ADJ"
            elif token in ["च", "वा", "अपि", "तु", "किन्तु", "परन्तु"]:
                tag = "CONJ"
            elif token in ["न", "मा", "इव", "एव"]:
                tag = "PART"
            elif token in ["अहम्", "त्वम्", "स:", "सा", "तत्", "अयम्", "इयम्", "इदम्"]:
                tag = "PRON"
            elif token.endswith(("त:", "तस्", "त्र", "दा")):
                tag = "ADV"
            elif re.match(r'^[१२३४५६७८९०]+$', token):
                tag = "NUM"
            else:
                # Default to noun for unknown - In a real implementation, this would be more sophisticated
                tag = "NOUN"
            
            tagged.append((token, self.pos_tags[tag]))
        
        return tagged
    
    def analyze_semantics(self, text):
        """
        Perform semantic analysis on Sanskrit text.
        
        Args:
            text (str): Sanskrit text to analyze
            
        Returns:
            dict: Dictionary containing semantic analysis results
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # This is a simplified semantic analysis
        # Real semantic analysis would involve deeper understanding of the text meaning
        
        # Basic entity recognition (simplified)
        entities = []
        for word in tokens["words"]:
            # Capitalize words might be proper nouns in some cases
            if word and word[0].isupper():
                entities.append({"word": word, "type": "Person/Place"})
        
        # Attempt to identify the main theme based on word frequency (very simplified)
        word_freq = {}
        for word in tokens["words"]:
            word_lower = word.lower()
            if len(word_lower) > 3:  # Ignore very short words
                word_freq[word_lower] = word_freq.get(word_lower, 0) + 1
        
        # Get top words by frequency
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Sentiment analysis (extremely simplified)
        # A real implementation would use a Sanskrit-specific sentiment lexicon or model
        positive_words = ["सुख", "आनन्द", "शान्ति", "प्रेम", "सत्य", "शुभ"]
        negative_words = ["दुःख", "क्रोध", "भय", "शोक", "हिंसा", "अशुभ"]
        
        sentiment = "neutral"
        pos_count = sum(1 for word in tokens["words"] if any(pos in word for pos in positive_words))
        neg_count = sum(1 for word in tokens["words"] if any(neg in word for neg in negative_words))
        
        if pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        
        return {
            "entities": entities,
            "top_words": [{"word": word, "frequency": freq} for word, freq in top_words],
            "sentiment": sentiment,
            "sentiment_stats": {
                "positive_words": pos_count,
                "negative_words": neg_count
            }
        }
        
    def learn_grammar_rule(self, instruction):
        """
        Learn a new Sanskrit grammar rule from a natural language instruction.
        For example: "iko yan aci" - "When 'ik' (i, ī, u, ū, ṛ, ṝ, ḷ, ḹ) is followed by 'ac' (vowel), 
        it is replaced by the corresponding 'yan' (y, v, r, l)"
        
        Args:
            instruction (str): Natural language instruction describing the rule
            
        Returns:
            dict: Information about the learned rule
        """
        # Basic rule patterns to recognize
        rule_patterns = [
            # "iko yan aci" pattern - change character from group1 to group2 when followed by group3
            # Example: "iko yan aci" means characters from 'ik' group change to corresponding 'yan' when followed by 'ac'
            r'([a-zA-Z_]+)(?:o|\(\))? ([a-zA-Z_]+)(?:i|\(\))? ([a-zA-Z_]+)',
            # Substitution pattern in Hindi/Sanskrit description
            r'([^\s]+) के स्थान पर ([^\s]+) हो जाता है ([^\s]+) के पारे होने पर',
            # Simple English pattern
            r'replace ([a-zA-Z_]+) with ([a-zA-Z_]+) when followed by ([a-zA-Z_]+)'
        ]
        
        # Try to extract rule components
        rule_found = False
        source_group = None
        target_group = None
        condition_group = None
        
        # Check each pattern
        for pattern in rule_patterns:
            match = re.search(pattern, instruction, re.IGNORECASE)
            if match:
                source_group = match.group(1).strip()
                target_group = match.group(2).strip()
                condition_group = match.group(3).strip()
                rule_found = True
                break
        
        # If no pattern matched, try to extract known group names
        if not rule_found:
            # Look for character group names in the instruction
            known_groups = list(self.char_groups.keys())
            found_groups = []
            
            for group in known_groups:
                if group in instruction:
                    found_groups.append(group)
                
            # If we found exactly 3 groups, assume they're in the right order
            if len(found_groups) == 3:
                source_group = found_groups[0]
                target_group = found_groups[1]
                condition_group = found_groups[2]
                rule_found = True
        
        # If we've identified a rule, process it
        if rule_found:
            # Look up the character groups
            source_chars = self.char_groups.get(source_group, [])
            target_chars = self.char_groups.get(target_group, [])
            condition_chars = self.char_groups.get(condition_group, [])
            
            # If any group wasn't found, try with different casing/variations
            if not source_chars and source_group is not None:
                for group, chars in self.char_groups.items():
                    if (isinstance(group, str) and isinstance(source_group, str) and 
                        (group.lower() == source_group.lower() or 
                         (hasattr(group, 'replace') and group.replace('_', '') == source_group.lower()))):
                        source_chars = chars
                        source_group = group
                        break
            
            if not target_chars and target_group is not None:
                for group, chars in self.char_groups.items():
                    if (isinstance(group, str) and isinstance(target_group, str) and 
                        (group.lower() == target_group.lower() or 
                         (hasattr(group, 'replace') and group.replace('_', '') == target_group.lower()))):
                        target_chars = chars
                        target_group = group
                        break
            
            if not condition_chars and condition_group is not None:
                for group, chars in self.char_groups.items():
                    if (isinstance(group, str) and isinstance(condition_group, str) and 
                        (group.lower() == condition_group.lower() or 
                         (hasattr(group, 'replace') and group.replace('_', '') == condition_group.lower()))):
                        condition_chars = chars
                        condition_group = group
                        break
            
            # If we have valid character groups, create the rule
            if source_chars and target_chars and condition_chars:
                # Map source characters to target characters 
                # (assuming they have corresponding positions)
                mapping = {}
                
                # Create mapping only for the available corresponding positions
                for i in range(min(len(source_chars), len(target_chars))):
                    src_char = source_chars[i]
                    tgt_char = target_chars[i]
                    
                    # For each condition character, create a rule
                    for cond_char in condition_chars:
                        rule_key = f"{src_char}+{cond_char}"
                        mapping[rule_key] = tgt_char + cond_char
                
                # Add these rules to our custom sandhi rules
                self.custom_sandhi_rules.update(mapping)
                
                # Save the learned rule
                rule_info = {
                    "name": f"{source_group}_{target_group}_{condition_group}",
                    "description": instruction,
                    "source_group": source_group,
                    "target_group": target_group,
                    "condition_group": condition_group,
                    "examples": [
                        {"input": f"{source_chars[0]}+{condition_chars[0]}", 
                         "output": f"{target_chars[0]}{condition_chars[0]}"}
                    ],
                    "rule_count": len(mapping)
                }
                
                self.learned_rules.append(rule_info)
                
                return {
                    "success": True,
                    "rule": rule_info,
                    "mappings": mapping
                }
            else:
                # We couldn't find all the character groups
                missing_groups = []
                if not source_chars:
                    missing_groups.append(source_group)
                if not target_chars:
                    missing_groups.append(target_group)
                if not condition_chars:
                    missing_groups.append(condition_group)
                    
                return {
                    "success": False,
                    "error": f"Could not find character groups: {', '.join(missing_groups)}",
                    "extracted_groups": {
                        "source_group": source_group,
                        "target_group": target_group,
                        "condition_group": condition_group
                    }
                }
        
        # We couldn't identify a rule pattern
        return {
            "success": False,
            "error": "Could not identify a grammar rule pattern in the instruction",
            "instruction": instruction
        }
    
    def apply_learned_rules(self, text):
        """
        Apply learned grammar rules to the given Sanskrit text.
        
        Args:
            text (str): The Sanskrit text to process
            
        Returns:
            dict: Results of applying the rules
        """
        if not self.custom_sandhi_rules:
            return {
                "success": False,
                "message": "No custom rules to apply",
                "text": text
            }
        
        # Tokenize the text
        tokens = self.tokenize(text)
        words = tokens["words"]
        
        # Apply rules to word boundaries
        changes = []
        result_words = []
        
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            
            if current_word and next_word:
                # Check if the joining of words matches any of our custom rules
                if current_word[-1] + "+" + next_word[0] in self.custom_sandhi_rules:
                    rule_key = current_word[-1] + "+" + next_word[0]
                    replacement = self.custom_sandhi_rules[rule_key]
                    
                    # Record the change
                    changes.append({
                        "position": i,
                        "original": f"{current_word} {next_word}",
                        "rule_applied": rule_key,
                        "result": f"{current_word[:-1]}{replacement}"
                    })
                    
                    # Apply the change
                    result_words.append(current_word[:-1] + replacement)
                    # Skip the next word as it's been combined
                    i += 1
                else:
                    result_words.append(current_word)
            else:
                result_words.append(current_word)
        
        # Add the last word if it wasn't processed
        if result_words and len(result_words) < len(words):
            result_words.append(words[-1])
        
        # Reconstruct the text
        result_text = " ".join(result_words)
        
        return {
            "success": True,
            "original_text": text,
            "processed_text": result_text,
            "changes": changes,
            "rules_applied": len(changes)
        }
    
    def get_learned_rules(self):
        """
        Get information about the learned grammar rules.
        
        Returns:
            list: List of learned rules
        """
        return self.learned_rules
