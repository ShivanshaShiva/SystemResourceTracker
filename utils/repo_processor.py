import os
import git
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import tiktoken
from git import Repo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

class RepoProcessor:
    """
    Class for downloading, processing, and extracting code from GitHub repositories
    to use as training data for code generation models.
    """
    
    def __init__(self, temp_dir="temp_repos"):
        """
        Initialize the RepoProcessor class.
        
        Args:
            temp_dir (str): Directory to temporarily store downloaded repositories
        """
        self.temp_dir = temp_dir
        self.repos = {}
        self.code_data = {}
        self.tokenizer = None
        self.vectorizer = None
        
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # File extensions to process for different languages
        self.language_extensions = {
            "Python": [".py"],
            "JavaScript": [".js", ".jsx"],
            "TypeScript": [".ts", ".tsx"],
            "Java": [".java"],
            "C#": [".cs"],
            "C++": [".cpp", ".hpp", ".h", ".cc"],
            "C": [".c", ".h"],
            "Go": [".go"],
            "Rust": [".rs"],
            "Ruby": [".rb"],
            "PHP": [".php"],
            "Swift": [".swift"],
            "Kotlin": [".kt", ".kts"]
        }
    
    def clone_repository(self, repo_url, language=None, depth=1):
        """
        Clone a GitHub repository to analyze its code.
        
        Args:
            repo_url (str): URL of the GitHub repository
            language (str, optional): Programming language to filter files for
            depth (int): Clone depth (1 for shallow clone)
            
        Returns:
            dict: Repository information
        """
        # Extract repo name from URL
        repo_name = repo_url.split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        
        repo_path = os.path.join(self.temp_dir, repo_name)
        
        # Clone the repository
        try:
            repo = Repo.clone_from(
                repo_url, 
                repo_path, 
                depth=depth,  # Shallow clone to save space and time
            )
            
            # Store repo information
            self.repos[repo_name] = {
                "path": repo_path,
                "url": repo_url,
                "language": language,
                "files_processed": 0,
                "code_snippets": 0
            }
            
            return {
                "success": True,
                "repo_name": repo_name,
                "message": f"Repository '{repo_name}' cloned successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error cloning repository: {str(e)}"
            }
    
    def process_repository(self, repo_name, max_files=None, min_lines=10, max_lines=1000):
        """
        Process the files in a cloned repository to extract code snippets.
        
        Args:
            repo_name (str): Name of the repository
            max_files (int, optional): Maximum number of files to process
            min_lines (int): Minimum number of lines for a code snippet
            max_lines (int): Maximum number of lines for a code snippet
            
        Returns:
            dict: Processing results
        """
        if repo_name not in self.repos:
            return {
                "success": False,
                "message": f"Repository '{repo_name}' not found. Clone it first."
            }
        
        repo_info = self.repos[repo_name]
        repo_path = repo_info["path"]
        language = repo_info["language"]
        
        # Determine which file extensions to look for
        extensions = self.language_extensions.get(language, []) if language else []
        
        # Initialize code data for this repo
        self.code_data[repo_name] = []
        
        # Walk through the repository files
        file_count = 0
        code_snippet_count = 0
        
        for root, dirs, files in os.walk(repo_path):
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')
            
            for file in files:
                # Check if we've reached the maximum files to process
                if max_files and file_count >= max_files:
                    break
                
                # Check if the file has a relevant extension
                if extensions and not any(file.endswith(ext) for ext in extensions):
                    continue
                
                try:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    
                    # Split code into lines and filter by length
                    lines = code.split('\n')
                    if len(lines) < min_lines or len(lines) > max_lines:
                        continue
                    
                    # Store the code snippet with metadata
                    rel_path = os.path.relpath(file_path, repo_path)
                    self.code_data[repo_name].append({
                        "filename": file,
                        "path": rel_path,
                        "language": language,
                        "code": code,
                        "line_count": len(lines)
                    })
                    
                    # Update counters
                    file_count += 1
                    code_snippet_count += 1
                    
                except Exception as e:
                    # Skip files that can't be read
                    continue
        
        # Update repository information
        self.repos[repo_name]["files_processed"] = file_count
        self.repos[repo_name]["code_snippets"] = code_snippet_count
        
        return {
            "success": True,
            "repo_name": repo_name,
            "files_processed": file_count,
            "code_snippets": code_snippet_count,
            "message": f"Processed {file_count} files, extracted {code_snippet_count} code snippets"
        }
    
    def cleanup_repository(self, repo_name):
        """
        Remove the cloned repository to free up space.
        
        Args:
            repo_name (str): Name of the repository
            
        Returns:
            dict: Cleanup results
        """
        if repo_name not in self.repos:
            return {
                "success": False,
                "message": f"Repository '{repo_name}' not found."
            }
        
        repo_path = self.repos[repo_name]["path"]
        
        try:
            shutil.rmtree(repo_path)
            return {
                "success": True,
                "message": f"Repository '{repo_name}' removed successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error removing repository: {str(e)}"
            }
    
    def cleanup_all_repositories(self):
        """
        Remove all cloned repositories.
        
        Returns:
            dict: Cleanup results
        """
        success_count = 0
        failure_count = 0
        
        for repo_name in list(self.repos.keys()):
            result = self.cleanup_repository(repo_name)
            if result["success"]:
                success_count += 1
                # Remove from repos dict
                self.repos.pop(repo_name, None)
            else:
                failure_count += 1
        
        return {
            "success": success_count > 0 and failure_count == 0,
            "success_count": success_count,
            "failure_count": failure_count,
            "message": f"Removed {success_count} repositories, {failure_count} failures"
        }
    
    def get_all_code_snippets(self, languages=None, min_lines=None, max_lines=None):
        """
        Get all code snippets from all processed repositories, with optional filtering.
        
        Args:
            languages (list, optional): Filter by programming languages
            min_lines (int, optional): Minimum number of lines
            max_lines (int, optional): Maximum number of lines
            
        Returns:
            list: Filtered code snippets
        """
        all_snippets = []
        
        for repo_name, snippets in self.code_data.items():
            for snippet in snippets:
                # Apply filters if provided
                if languages and snippet["language"] not in languages:
                    continue
                    
                if min_lines and snippet["line_count"] < min_lines:
                    continue
                    
                if max_lines and snippet["line_count"] > max_lines:
                    continue
                
                all_snippets.append(snippet)
        
        return all_snippets
    
    def export_to_dataframe(self, languages=None, min_lines=None, max_lines=None):
        """
        Export code snippets to a pandas DataFrame for analysis or training.
        
        Args:
            languages (list, optional): Filter by programming languages
            min_lines (int, optional): Minimum number of lines
            max_lines (int, optional): Maximum number of lines
            
        Returns:
            DataFrame: Code snippets in DataFrame format
        """
        snippets = self.get_all_code_snippets(languages, min_lines, max_lines)
        return pd.DataFrame(snippets)
    
    def export_to_json(self, output_path, languages=None, min_lines=None, max_lines=None):
        """
        Export code snippets to a JSON file.
        
        Args:
            output_path (str): Path to save the JSON file
            languages (list, optional): Filter by programming languages
            min_lines (int, optional): Minimum number of lines
            max_lines (int, optional): Maximum number of lines
            
        Returns:
            dict: Export results
        """
        snippets = self.get_all_code_snippets(languages, min_lines, max_lines)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(snippets, f, indent=2)
            
            return {
                "success": True,
                "snippets_exported": len(snippets),
                "message": f"Exported {len(snippets)} code snippets to {output_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error exporting to JSON: {str(e)}"
            }
    
    def prepare_training_data(self, languages=None, test_size=0.2, random_state=42):
        """
        Prepare training and testing datasets for code generation model training.
        
        Args:
            languages (list, optional): Filter by programming languages
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Training and testing data
        """
        # Get all relevant code snippets
        snippets = self.get_all_code_snippets(languages)
        
        if not snippets:
            return {
                "success": False,
                "message": "No code snippets found matching the criteria"
            }
        
        # Extract code and create labels (in this case, the language)
        codes = [s["code"] for s in snippets]
        languages = [s["language"] for s in snippets]
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            codes, languages, test_size=test_size, random_state=random_state
        )
        
        return {
            "success": True,
            "train_data": {"code": X_train, "language": y_train},
            "test_data": {"code": X_test, "language": y_test},
            "total_snippets": len(snippets),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }
    
    def initialize_vectorizer(self, snippets=None):
        """
        Initialize a TF-IDF vectorizer for code similarity analysis.
        
        Args:
            snippets (list, optional): Code snippets to fit the vectorizer
            
        Returns:
            TfidfVectorizer: Fitted vectorizer
        """
        if not snippets:
            snippets = self.get_all_code_snippets()
            
        if not snippets:
            return None
        
        # Extract code text
        code_texts = [s["code"] for s in snippets]
        
        # Initialize and fit the vectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            max_features=5000,
            stop_words='english'
        )
        
        self.vectorizer.fit(code_texts)
        return self.vectorizer
    
    def find_similar_code(self, query_code, top_n=5):
        """
        Find similar code snippets to the query code.
        
        Args:
            query_code (str): Code to find similar snippets for
            top_n (int): Number of similar snippets to return
            
        Returns:
            list: Similar code snippets with similarity scores
        """
        if not self.vectorizer:
            snippets = self.get_all_code_snippets()
            self.initialize_vectorizer(snippets)
            
        if not self.vectorizer:
            return []
        
        # Get all snippets
        snippets = self.get_all_code_snippets()
        code_texts = [s["code"] for s in snippets]
        
        # Vectorize the query code
        query_vector = self.vectorizer.transform([query_code])
        
        # Vectorize all code snippets
        all_vectors = self.vectorizer.transform(code_texts)
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, all_vectors)[0]
        
        # Get the indices of the top similar snippets
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        # Prepare results
        results = []
        for i in top_indices:
            results.append({
                "snippet": snippets[i],
                "similarity_score": similarities[i]
            })
        
        return results
    
    def generate_code_completion(self, partial_code, language, max_candidates=5, min_similarity=0.3):
        """
        Generate code completion based on similar snippets in the repository.
        
        Args:
            partial_code (str): Partial code to complete
            language (str): Programming language
            max_candidates (int): Maximum number of completion candidates
            min_similarity (float): Minimum similarity score threshold
            
        Returns:
            list: Candidate code completions
        """
        # Find similar code snippets
        similar_snippets = self.find_similar_code(partial_code, top_n=10)
        
        # Filter by language and similarity threshold
        filtered_snippets = [
            s for s in similar_snippets 
            if s["snippet"]["language"] == language and s["similarity_score"] >= min_similarity
        ]
        
        # Limit to max_candidates
        candidates = filtered_snippets[:max_candidates]
        
        # Extract the code as completion candidates
        completion_candidates = [{"code": c["snippet"]["code"], "score": c["similarity_score"]} for c in candidates]
        
        return completion_candidates
    
    def summarize_repositories(self):
        """
        Summarize information about all processed repositories.
        
        Returns:
            dict: Repository statistics
        """
        total_repos = len(self.repos)
        total_files = sum(repo["files_processed"] for repo in self.repos.values())
        total_snippets = sum(repo["code_snippets"] for repo in self.repos.values())
        
        languages = {}
        for repo in self.repos.values():
            lang = repo["language"]
            if lang:
                languages[lang] = languages.get(lang, 0) + repo["code_snippets"]
        
        return {
            "total_repositories": total_repos,
            "total_files_processed": total_files,
            "total_code_snippets": total_snippets,
            "languages": languages
        }