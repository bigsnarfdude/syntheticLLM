import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import Dict

class RewardCalculator:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.toxicity = pipeline("text-classification", 
                               model="unitary/toxic-bert")
        
    def calculate_rewards(self, row: Dict) -> Dict:
        """Calculate multiple reward signals for a conversation pair"""
        user_content = row['user_content']
        assistant_content = row['assistant_content']
        
        # Get embeddings
        embeddings = self.embedder.encode([user_content, assistant_content])
        
        # Calculate rewards
        return {
            'semantic_coherence': np.dot(embeddings[0], embeddings[1]),
            'safety_score': 1 - self.toxicity(assistant_content)[0]['score'],
            'conciseness': 1 / (1 + len(assistant_content)/100)
        }
    
    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add reward columns to dataframe"""
        # Pivot to conversation pairs
        conv_pairs = df.pivot(index='ID', columns='Role', values='Content')
        
        # Calculate rewards
        rewards = conv_pairs.apply(
            lambda x: self.calculate_rewards({
                'user_content': x['user'],
                'assistant_content': x['assistant']
            }), 
            axis=1
        ).apply(pd.Series)
        
        return df.merge(rewards, left_on='ID', right_index=True)