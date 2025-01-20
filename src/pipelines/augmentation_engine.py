import requests
import json
import pandas as pd
import logging

class OllamaAugmentationEngine:
    def __init__(self, 
                 model="phi4:lastest",
                 host="http://localhost:11434",
                 temperature=0.7,
                 max_tokens=256):
        """
        Initialize Ollama-based augmentation engine
        
        Args:
            model (str): Ollama model to use for augmentation
            host (str): Ollama API host
            temperature (float): Sampling temperature for text generation
            max_tokens (int): Maximum tokens to generate
        """
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Augmentation prompt template
        self.prompt_template = """You are a text augmentation assistant. 
Paraphrase the following text while preserving its core meaning and intent. 
Ensure the paraphrased version is semantically equivalent but uses different wording:

Original Text: {text}

Paraphrased Version:"""

    def _generate_variant(self, text):
        """
        Generate a text variant using Ollama
        
        Args:
            text (str): Input text to augment
        
        Returns:
            str: Augmented text variant
        """
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": self.prompt_template.format(text=text),
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
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
            return text

    def augment_text(self, text: str, num_variants: int = 2) -> list:
        """
        Generate multiple variants of input text
        
        Args:
            text (str): Input text to augment
            num_variants (int): Number of variants to generate
        
        Returns:
            list: Augmented text variants
        """
        variants = []
        for _ in range(num_variants):
            variant = self._generate_variant(text)
            variants.append(variant)
        return variants

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create augmented dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Dataframe with augmented entries
        """
        augmented = []
        for _, row in df.iterrows():
            # Augment user messages
            if row.get('role') == 'prompter':
                variants = self.augment_text(row['text'])
                for var in variants:
                    new_row = row.copy()
                    new_row['text'] = var
                    new_row['is_augmented'] = True
                    augmented.append(new_row)
        
        # Combine original and augmented data
        return pd.concat([df, pd.DataFrame(augmented)], ignore_index=True)

# Example usage
if __name__ == "__main__":
    # Demonstrate augmentation
    augmenter = OllamaAugmentationEngine(model="llama2")
    
    # Single text augmentation
    text = "Explain the concept of monopsony in economics."
    variants = augmenter.augment_text(text)
    
    print("Original Text:", text)
    print("Variants:")
    for variant in variants:
        print("- ", variant)
