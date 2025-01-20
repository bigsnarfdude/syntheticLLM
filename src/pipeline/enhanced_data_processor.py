import pandas as pd
import numpy as np
import re
import requests
import json
import logging

class OllamaEnhancedDataProcessor:
    def __init__(self, 
                 model="llama2",
                 host="http://localhost:11434",
                 quality_threshold: float = 0.7, 
                 toxicity_threshold: float = 0.1):
        """
        Initialize Ollama-enhanced data processing pipeline
        
        Args:
            model (str): Ollama model to use
            host (str): Ollama API host
            quality_threshold (float): Minimum quality score to keep
            toxicity_threshold (float): Maximum toxicity score to allow
        """
        self.model = model
        self.host = host
        self.quality_threshold = quality_threshold
        self.toxicity_threshold = toxicity_threshold
        
        # Predefined intent categories
        self.intent_categories = [
            'information_request', 
            'explanation', 
            'comparison', 
            'problem_solving', 
            'creative', 
            'opinion'
        ]

    def _ollama_generate(self, prompt: str) -> str:
        """
        Generate text using Ollama API
        
        Args:
            prompt (str): Input prompt
        
        Returns:
            str: Generated text
        """
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100
                    }
                }
            )
            response.raise_for_status()
            
            # Aggregate response chunks
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        full_response += chunk['response']
            
            return full_response.strip()
        
        except requests.RequestException as e:
            logging.error(f"Ollama API error: {e}")
            return ""

    def classify_intent_with_ollama(self, text: str) -> str:
        """
        Classify intent using Ollama model
        
        Args:
            text (str): Input text
        
        Returns:
            str: Predicted intent
        """
        prompt = f"""Classify the intent of the following text into one of these categories:
{', '.join(self.intent_categories)}

Text: {text}

Intent:"""
        
        intent_response = self._ollama_generate(prompt)
        
        # Clean and validate the response
        for category in self.intent_categories:
            if category.lower() in intent_response.lower():
                return category
        
        return 'unknown'

    def calculate_text_complexity_with_ollama(self, text: str) -> float:
        """
        Calculate text complexity using Ollama model
        
        Args:
            text (str): Input text
        
        Returns:
            float: Complexity score
        """
        prompt = f"""Analyze the complexity of the following text. 
Provide a complexity score from 0 to 1, where:
0 = Very Simple
0.5 = Moderately Complex
1 = Highly Complex

Consider factors like:
- Vocabulary sophistication
- Sentence structure
- Conceptual depth
- Technical terminology

Text: {text}

Complexity Score (0-1):"""
        
        try:
            complexity_response = self._ollama_generate(prompt)
            
            # Extract numeric score
            score = re.findall(r'(\d+\.?\d*)', complexity_response)
            
            if score:
                # Normalize to 0-1 range
                return min(max(float(score[0]) / 1.0, 0), 1)
            
            return 0.5  # Default to middle complexity
        
        except Exception:
            return 0.5

    def filter_high_quality_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter conversations based on Ollama-assisted quality assessment
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        # Add quality assessment column
        df['ollama_complexity'] = df['text'].apply(self.calculate_text_complexity_with_ollama)
        
        # Apply filtering
        filtered_df = df[
            (df['ollama_complexity'] >= self.quality_threshold)
        ]
        
        return filtered_df

    def enrich_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich metadata with Ollama-generated insights
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Dataframe with additional metadata
        """
        # Calculate Ollama-based complexity
        df['ollama_complexity'] = df['text'].apply(self.calculate_text_complexity_with_ollama)
        
        # Existing metadata calculations
        df['conversation_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        return df

    def classify_intents(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify conversation intents using Ollama
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Dataframe with predicted intents
        """
        # Predict intents using Ollama
        df['predicted_intent'] = df['text'].apply(self.classify_intent_with_ollama)
        
        return df

    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all processing steps
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Processed dataframe
        """
        # Apply steps in sequence
        df = self.filter_high_quality_data(df)
        df = self.enrich_metadata(df)
        df = self.classify_intents(df)
        
        return df

# Example usage
if __name__ == "__main__":
    # Create sample dataframe
    sample_data = pd.DataFrame({
        'text': [
            "What is the capital of France?",
            "Explain the theory of relativity in simple terms.",
            "I want to learn about quantum computing.",
            "Write a short story about a robot's adventure."
        ]
    })
    
    # Initialize and process data
    processor = OllamaEnhancedDataProcessor()
    processed_df = processor.process_dataset(sample_data)
    
    # Display results
    print(processed_df.to_string())
    
    # Optional: Save processed data
    processed_df.to_csv('ollama_processed_conversations.csv', index=False)
