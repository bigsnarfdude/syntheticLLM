import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import pandas as pd
import requests

@dataclass
class ContentLabels:
    relevance: float
    accuracy: float
    completeness: float
    clarity: float
    depth: float

@dataclass
class ToneLabels:
    sentiment: float
    toxicity: float
    sarcasm: float
    humor: float
    formality: float

@dataclass
class SafetyLabels:
    hate_speech: float
    threat: float
    misinformation: float
    manipulation: float
    pii: float

@dataclass
class StyleLabels:
    creativity: float
    engagement: float
    eloquence: float

@dataclass
class TextLabels:
    content: ContentLabels
    tone: ToneLabels
    safety: SafetyLabels
    style: StyleLabels

@dataclass
class PipelineConfig:
    input_path: Path
    output_path: Path
    steps: List[str]
    log_level: str = "INFO"

class PipelineComponent:
    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
        
    def process_single(self, row: pd.Series) -> pd.Series:
        return row

class OllamaAugmentationEngine(PipelineComponent):
    def __init__(
        self, 
        model: str = "phi4:latest",
        host: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 256,
        num_variants: int = 2
    ):
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_variants = num_variants
        self.logger = logging.getLogger(__name__)
        
        self.prompt_template = """Rewrite the following text while preserving its core meaning and intent:

Original Text: {text}

Paraphrased Version:"""

    def _generate_variant(self, text: str) -> Optional[str]:
        if not text or not isinstance(text, str):
            return None

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
                },
                timeout=30
            )
            response.raise_for_status()
            
            full_response = "".join(
                chunk.get('response', '') 
                for chunk in (json.loads(line) for line in response.iter_lines() if line)
                if 'response' in chunk
            ).strip()
            
            return full_response if full_response and full_response != text else None
        
        except Exception as e:
            self.logger.error(f"Variant generation error: {e}")
            return None

    def process_single(self, row: pd.Series) -> pd.Series:
        if row.get('role') == 'prompter' and row.get('text'):
            variants = [
                variant for variant in 
                (self._generate_variant(row['text']) for _ in range(self.num_variants)) 
                if variant
            ]
            for i, var in enumerate(variants, 1):
                row[f'augmented_{i}'] = var
        return row

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            self.logger.info("Empty DataFrame. Skipping augmentation.")
            return df
        
        augmentable_rows = df[df['role'].isin(['prompter', 'assistant'])]
        self.logger.info(f"Found {len(augmentable_rows)} rows to augment")
        
        processed_rows = []
        for idx, row in tqdm(augmentable_rows.iterrows(), total=len(augmentable_rows), desc="Augmenting data"):
            processed_row = self.process_single(row)
            processed_rows.append(processed_row)
            
        augmented_df = pd.concat([df] + processed_rows, ignore_index=True)
        
        self.logger.info(
            f"Augmentation complete. Original: {len(df)}, "
            f"Augmented: {len(processed_rows)}, Total: {len(augmented_df)}"
        )
        
        return augmented_df

class LLMLabeler(PipelineComponent):
    def __init__(
        self,
        model: str = "phi4:latest",
        host: str = "http://localhost:11434",
        temperature: float = 0.3
    ):
        self.model = model
        self.host = host
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)
        
        self.prompt_template = """Analyze the following text and provide scores for each category on specified scales.
Be strict and objective in scoring. Provide brief justification for each score.

Text: {text}

Score format (each on new line):
category: score [scale] - brief justification

Categories to score:
Content:
- Relevance (0-1)
- Accuracy (0-1) 
- Completeness (0-1)
- Clarity (0-1)
- Depth (0-1)

Tone:
- Sentiment (-1 to 1)
- Toxicity (0-1)
- Sarcasm (0-1)
- Humor (0-1)
- Formality (0-1)

Safety:
- Hate speech (0-1)
- Threat level (0-1)
- Misinformation (0-1)
- Manipulation (0-1)
- PII exposure (0-1)

Style:
- Creativity (0-1)
- Engagement (0-1)
- Eloquence (0-1)"""

    def _analyze_text(self, text: str) -> Optional[TextLabels]:
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": self.prompt_template.format(text=text),
                    "options": {"temperature": self.temperature}
                },
                timeout=30
            )
            response.raise_for_status()
            
            full_response = "".join(
                chunk.get('response', '') 
                for chunk in (json.loads(line) for line in response.iter_lines() if line)
                if 'response' in chunk
            ).strip()
            
            scores = self._parse_llm_response(full_response)
            return TextLabels(**scores)
            
        except Exception as e:
            self.logger.error(f"Text analysis error: {e}")
            return None

    def _parse_llm_response(self, response_text: str) -> Dict:
        scores = {
            'content': {},
            'tone': {},
            'safety': {},
            'style': {}
        }
        
        current_category = None
        for line in response_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.lower() in ['content:', 'tone:', 'safety:', 'style:']:
                current_category = line.lower().rstrip(':')
                continue
                
            if current_category and ':' in line:
                label, rest = line.split(':', 1)
                label = label.strip().lower().replace(' ', '_')
                
                # Extract score from the format "score [scale] - justification"
                score_part = rest.split('-')[0].strip()
                score = float(score_part.split('[')[0].strip())
                
                if current_category in scores:
                    scores[current_category][label] = score
        
        return {
            'content': ContentLabels(**scores['content']),
            'tone': ToneLabels(**scores['tone']),
            'safety': SafetyLabels(**scores['safety']),
            'style': StyleLabels(**scores['style'])
        }

    def process_single(self, row: pd.Series) -> pd.Series:
        if row.get('text'):
            labels = self._analyze_text(row['text'])
            if labels:
                row['content_labels'] = asdict(labels.content)
                row['tone_labels'] = asdict(labels.tone)
                row['safety_labels'] = asdict(labels.safety)
                row['style_labels'] = asdict(labels.style)
        return row

    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            self.logger.info("Empty DataFrame. Skipping labeling.")
            return df
        
        self.logger.info(f"Starting labeling for {len(df)} rows")
        
        processed_df = df.copy()
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Labeling data"):
            processed_df.loc[idx] = self.process_single(row)
            
        self.logger.info("Labeling complete")
        return processed_df

class DataPipeline:
    VALID_COMPONENTS: Dict[str, Any] = {
        'augment': OllamaAugmentationEngine,
        'label': LLMLabeler
    }

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._setup_logging()
        self._validate_paths()
        self.steps = self._validate_steps()

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _validate_paths(self) -> None:
        if not self.config.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_path}")
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _validate_steps(self) -> List[str]:
        return [step for step in self.config.steps if step in self.VALID_COMPONENTS]

    def _parse_jsonl(self) -> pd.DataFrame:
        conversations = []
        with open(self.config.input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    conversations.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    self.logger.warning(f"Skipping corrupted JSON on line {line_num}")
        
        df = pd.DataFrame(conversations)
        self.logger.info(f"Loaded {len(df)} entries with columns: {', '.join(df.columns)}")
        return df

    def _initialize_components(self) -> Dict[str, PipelineComponent]:
        return {
            step: self.VALID_COMPONENTS[step]() 
            for step in self.steps
        }

    def run(self) -> None:
        try:
            df = self._parse_jsonl()
            components = self._initialize_components()

            for step, component in components.items():
                self.logger.info(f"Processing {step}")
                df = component.process_batch(df)

            df.to_json(self.config.output_path, orient='records', lines=True)
            self.logger.info(f"Pipeline completed. Output saved to {self.config.output_path}")

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise

def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Data augmentation and labeling pipeline")
    parser.add_argument('--input', type=str, required=True, help="Input JSONL file path")
    parser.add_argument('--output', type=str, required=True, help="Output file path")
    parser.add_argument('--steps', nargs='+', default=['augment', 'label'],
                       help="Processing steps to apply")
    parser.add_argument('--log-level', type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    return PipelineConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        steps=args.steps,
        log_level=args.log_level
    )

if __name__ == "__main__":
    config = parse_args()
    pipeline = DataPipeline(config)
    pipeline.run()
