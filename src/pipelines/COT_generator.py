import pandas as pd
import requests
import json
from typing import Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CoTConfig:
    model: str = "phi4:latest"
    temperature: float = 0.7
    max_tokens: int = 256
    ollama_host: str = "http://localhost:11434"

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
            # Format for Ollama API
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
                stream=True  # Enable streaming response
            )
            response.raise_for_status()
            
            # Process streaming response
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
        """Add CoT column to dataframe"""
        # Group messages by message_tree_id to pair prompts with responses
        conversations = []
        for tree_id, group in df.groupby('message_tree_id'):
            prompter_msg = group[group['role'] == 'prompter']['text'].iloc[0] if any(group['role'] == 'prompter') else ''
            assistant_msg = group[group['role'] == 'assistant']['text'].iloc[0] if any(group['role'] == 'assistant') else ''
            
            if prompter_msg and assistant_msg:
                cot = self.generate_cot(prompter_msg, assistant_msg)
                # Add CoT to both messages in the conversation
                for idx in group.index:
                    df.at[idx, 'cot_steps'] = cot
        
        return df