import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from config.settings import settings
from sklearn.model_selection import train_test_split
from typing import Tuple

class PCAFraudDetector:
    def __init__(self):
        self.pca_transformer = None
        self.model = None

    def fit(self, data: pd.DataFrame):
        """Train both PCA transformer and fraud detection model"""
        from src.features.pca_transformer import PCATransformer
        from src.models.model import FraudDetectionModel
        
        # 1. Prepare data
        X = data.drop('is_fraud', axis=1)
        y = data['is_fraud']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Store feature names
        self.feature_names_in_ = list(X.columns)
        
        # 2. Fit PCA transformer
        self.pca_transformer = PCATransformer()
        self.pca_transformer.fit(X_train)
        X_train_pca = self.pca_transformer.transform(X_train)
        X_val_pca = self.pca_transformer.transform(X_val)
        
        # 3. Train fraud detection model
        self.model = FraudDetectionModel()
        self.model.train_ensemble(X_train_pca, y_train, X_val_pca, y_val)

    def predict(self, raw_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data"""
        if not self.pca_transformer or not self.model:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Ensure columns match training data
        raw_data = raw_data[self.feature_names_in_]
        
        # Transform and predict
        pca_features = self.pca_transformer.transform(raw_data)
        pred_class = self.model.predict(pca_features)
        pred_proba = self.model.predict_proba(pca_features)[:, 1]
        
        return pred_class, pred_proba

    def save(self):
        """Save both components to disk"""
        if not self.pca_transformer or not self.model:
            raise ValueError("Model not trained. Call fit() first.")
            
        settings.MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump({
            'pca_transformer': self.pca_transformer,
            'model': self.model,
            'feature_names_in_': self.feature_names_in_
        }, settings.MODEL_DIR / "pca_fraud_detector.joblib")

    @classmethod
    def load(self, model_path: str):
        """Load the trained PCA transformer and model"""
        import joblib
        try:
            data = joblib.load(model_path)
            self.pca_transformer = data['pca_transformer']
            self.model = data['model']
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")