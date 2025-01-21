import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from typing import Dict
from tqdm.auto import tqdm

class RewardCalculator:
    def __init__(self):
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        self.toxicity = pipeline("text-classification", 
                               model="unitary/toxic-bert")
        
    def calculate_rewards(self, row: Dict) -> Dict:
        """Calculate multiple reward signals for a conversation pair"""
        user_content = row['user_text']
        assistant_content = row['assistant_text']
        
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
        # Group conversations by message_tree_id and role
        conv_pairs = df.pivot_table(
            index='message_tree_id',
            columns='role',
            values='text',
            aggfunc='first'
        ).rename(columns={
            'prompter': 'user_text',
            'assistant': 'assistant_text'
        })
        
        # Calculate rewards for complete conversation pairs
        valid_pairs = conv_pairs.dropna()
        rewards = []
        for idx, row in tqdm(valid_pairs.iterrows(), total=len(valid_pairs), desc="Calculating rewards"):
            reward = self.calculate_rewards(row)
            rewards.append({**reward, 'message_tree_id': idx})
        rewards = pd.DataFrame(rewards)
        
        # Merge rewards back to original dataframe
        return df.merge(
            rewards.reset_index(), 
            on='message_tree_id', 
            how='left'
        )