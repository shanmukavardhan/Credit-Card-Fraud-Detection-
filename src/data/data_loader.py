# data_loader.py
import pandas as pd
from pathlib import Path
from typing import Tuple
from config.settings import settings

class DataLoader:
    @staticmethod
    def load_raw_data() -> pd.DataFrame:
        """Load raw credit card transaction data"""
        return pd.read_csv(settings.RAW_DATA_PATH)
    
    @staticmethod
    def save_processed_data(df: pd.DataFrame, filename: str) -> None:
        """Save processed data to disk"""
        filepath = settings.PROCESSED_DATA_DIR / filename
        df.to_csv(filepath, index=False)
        
    @staticmethod
    def load_processed_data(filename: str) -> pd.DataFrame:
        """Load processed data from disk"""
        filepath = settings.PROCESSED_DATA_DIR / filename
        return pd.read_csv(filepath)