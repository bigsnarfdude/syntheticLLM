import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Type

import pandas as pd

from src.pipelines.COT_generator import CoTGenerator
from src.pipelines.reward_calculator import RewardCalculator
from src.pipelines.augmentation_engine import OllamaAugmentationEngine
from utils.data_utils import save_data

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
        """Optional method for row-by-row processing with progress tracking"""
        return row

class DataPipeline:
    VALID_COMPONENTS: Dict[str, Type[PipelineComponent]] = {
        'cot': CoTGenerator,
        'rewards': RewardCalculator,
        'augment': OllamaAugmentationEngine
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
            total_rows = len(df)

            for step, component in components.items():
                self.logger.info(f"Processing {step}")
                try:
                    if step == 'augment':
                        from tqdm import tqdm
                        processed_rows = []
                        for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Augmenting data"):
                            processed_row = component.process_single(row)
                            processed_rows.append(processed_row)
                        df = pd.concat(processed_rows, axis=1).T
                    else:
                        df = component.process_batch(df)
                except Exception as e:
                    self.logger.error(f"Error in {step}: {e}")
                    raise

            save_data(df, self.config.output_path)
            self.logger.info(f"Pipeline completed. Output saved to {self.config.output_path}")

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise

def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Synthetic data generation pipeline")
    parser.add_argument('--input', type=str, required=True, help="Input JSONL file path")
    parser.add_argument('--output', type=str, required=True, help="Output file path")
    parser.add_argument('--steps', nargs='+', default=['cot', 'rewards', 'augment'],
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