import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load raw conversation data"""
    path = Path(file_path)
    if path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif path.suffix == '.parquet':
        return pd.read_parquet(file_path)
    raise ValueError(f"Unsupported file format: {path.suffix}")

def save_data(df: pd.DataFrame, file_path: str):
    """Save processed data"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.csv':
        df.to_csv(file_path, index=False)
    elif path.suffix == '.parquet':
        df.to_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    logger.info(f"Saved processed data to {file_path}")