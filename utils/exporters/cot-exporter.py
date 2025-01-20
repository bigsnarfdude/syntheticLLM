import json
from pathlib import Path
import pandas as pd
from datetime import datetime

class CoTExporter:
    def __init__(self, output_dir="data/processed/with_cot"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(self, df: pd.DataFrame):
        """Export CoT data in human-readable JSON format"""
        # Format data as list of conversation dictionaries
        cot_data = []
        for _, row in df.iterrows():
            conversation = {
                "id": row.get("ID", ""),
                "timestamp": datetime.now().isoformat(),
                "user_query": row.get("user", ""),
                "assistant_response": row.get("assistant", ""),
                "chain_of_thought": row.get("cot_steps", "").split("\n"),
                "metadata": {
                    "labels": row.get("labels", {}),
                    "complexity": row.get("ollama_complexity", 0)
                }
            }
            cot_data.append(conversation)

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"cot_data_{timestamp}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(cot_data, f, indent=2, ensure_ascii=False)
            
        return output_file