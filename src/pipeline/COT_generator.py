# pipelines/cot_generation.py
import pandas as pd
import requests
from typing import Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CoTConfig:
    model: str = "llama2"  # Default local model
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
                }
            )
            response.raise_for_status()
            return self._parse_ollama_response(response.json())
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return ""

    def _parse_ollama_response(self, response_data: Dict) -> str:
        """Extract and format the CoT steps from Ollama response"""
        try:
            full_response = "".join(
                chunk.get("message", {}).get("content", "")
                for chunk in response_data.get("response", [])
            )
            return full_response.strip()
        except Exception as e:
            logger.error(f"Failed to parse Ollama response: {e}")
            return ""

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add CoT column to dataframe (same as before)"""
        df['cot_steps'] = df.apply(
            lambda row: self.generate_cot(
                row['Content'] if row['Role'] == 'user' else '',
                row['Content'] if row['Role'] == 'assistant' else ''
            ),
            axis=1
        )
        return df