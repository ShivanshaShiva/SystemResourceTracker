import re
import string
import random

class SanskritNLP:
    """
    Class for handling Sanskrit Natural Language Processing tasks.
    This includes tokenization, grammar analysis, POS tagging, and semantic analysis.
    """
    
    def __init__(self):
        # Load Sanskrit-specific stopwords and other resources
        self.sandhi_rules = self._initialize_sandhi_rules()
        self.pos_tags = self._initialize_pos_tags()
        self.sanskrit_chars = self._initialize_sanskrit_chars()
    
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
