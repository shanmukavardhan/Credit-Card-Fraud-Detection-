# data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
from config.settings import settings

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    # Ensure the correct column name is used
        target_column = 'Class'  # Update this if the column name is different
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the dataset")

        X = df.drop(target_column, axis=1)
        y = df[target_column]
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                  val_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into train, validation and test sets"""
        # First split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Then split train+val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test