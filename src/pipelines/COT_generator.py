# pipelines/cot_generation.py
import pandas as pd
import requests
import json
from typing import Dict, Optional
from dataclasses import dataclass
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

@dataclass
class CoTConfig:
    model: str = "phi4:latest"
    temperature: float = 0.7
    max_tokens: int = 256
    ollama_host: str = "http://localhost:11434"
    batch_size: int = 10

class CoTGenerator:
    def __init__(self, config: CoTConfig = CoTConfig()):
        self.config = config
        self.cot_prompt = """Generate concise reasoning steps that connect this query to the response:
Query: {query}
Response: {response}

Chain of Thought:
1."""
        
    def generate_cot(self, query: str, response: str) -> str:
        try:
            response = requests.post(
                f"{self.config.ollama_host}/api/chat",
                json={
                    "model": self.config.model,
                    "messages": [{
                        "role": "user",
                        "content": self.cot_prompt.format(
                            query=query, 
                            response=response
                        )
                    }],
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens
                    }
                },
                stream=True
            )
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        json_response = json.loads(line)
                        if 'message' in json_response:
                            content = json_response['message'].get('content', '')
                            if content:
                                full_response += content
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON line: {e}")
                        continue
                        
            return full_response.strip()
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return ""

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        # Get unique conversation threads
        tree_ids = df['message_tree_id'].unique()
        total_conversations = len(tree_ids)
        
        logger.info(f"Processing {total_conversations} conversations...")
        
        # Initialize progress bar for conversation threads
        for tree_id in tqdm(tree_ids, desc="Generating Chain-of-Thought", unit="conversation"):
            group = df[df['message_tree_id'] == tree_id]
            prompter_msg = group[group['role'] == 'prompter']['text'].iloc[0] if any(group['role'] == 'prompter') else ''
            assistant_msg = group[group['role'] == 'assistant']['text'].iloc[0] if any(group['role'] == 'assistant') else ''
            
            if prompter_msg and assistant_msg:
                cot = self.generate_cot(prompter_msg, assistant_msg)
                # Add CoT to all messages in the conversation
                df.loc[df['message_tree_id'] == tree_id, 'cot_steps'] = cot
        
        logger.info("Chain-of-Thought generation completed")
        return df