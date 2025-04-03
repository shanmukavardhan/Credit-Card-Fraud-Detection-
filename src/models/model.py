import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from typing import Tuple, List, Dict, Any
from config.settings import settings

class FraudDetectionModel:
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        
    def create_mlp_model(self, input_dim: int, neurons: List[int] = [128, 64], 
                        dropout_rate: float = 0.3, learning_rate: float = 0.001) -> Sequential:
        """Create a simple MLP model"""
        model = Sequential([
            Dense(neurons[0], activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(neurons[1], activation='relu'),
            BatchNormalization(),
            Dropout(dropout_rate),
            
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_random_forest(self, n_estimators: int = 100, class_weight: str = 'balanced') -> RandomForestClassifier:
        """Create a Random Forest classifier"""
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            class_weight=class_weight,
            n_jobs=-1
        )
    
    def create_xgboost(self, n_estimators: int = 100, learning_rate: float = 0.1, 
                      max_depth: int = 5, scale_pos_weight: float = None) -> XGBClassifier:
        """Create an XGBoost classifier"""
        return XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    def create_lightgbm(self, n_estimators: int = 100, learning_rate: float = 0.1, 
                       num_leaves: int = 31, class_weight: str = 'balanced') -> lgb.LGBMClassifier:
        """Create a LightGBM classifier"""
        return lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            class_weight=class_weight,
            random_state=42
        )
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Train an ensemble of models"""
        input_dim = X_train.shape[1]
        
        # Train MLP
        mlp_model = self.create_mlp_model(input_dim)
        mlp_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=256,
            verbose=1
        )
        self.models['mlp'] = mlp_model
        
        # Train Random Forest
        rf_model = self.create_random_forest()
        rf_model.fit(X_train, y_train)
        self.models['rf'] = rf_model
        
        # Train XGBoost
        scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
        xgb_model = self.create_xgboost(scale_pos_weight=scale_pos_weight)
        xgb_model.fit(X_train, y_train)
        self.models['xgb'] = xgb_model
        
        # Train LightGBM
        lgb_model = self.create_lightgbm()
        lgb_model.fit(X_train, y_train)
        self.models['lgb'] = lgb_model
        
        # Initialize weights (can be optimized later)
        self.model_weights = {
            'mlp': 0.25,
            'rf': 0.25,
            'xgb': 0.25,
            'lgb': 0.25
        }
    
    def predict(self, X: pd.DataFrame, threshold: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the ensemble"""
        if threshold is None:
            threshold = settings.THRESHOLD
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                predictions[name] = model.predict_proba(X)[:, 1]
            else:
                predictions[name] = model.predict(X).ravel()
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions['mlp'])
        for name, pred in predictions.items():
            ensemble_pred += self.model_weights[name] * pred
        
        # Apply threshold
        ensemble_class = (ensemble_pred > threshold).astype(int)
        
        return ensemble_class, ensemble_pred
    
    def save(self, filepath: str = None) -> None:
        """Save the model to disk"""
        if filepath is None:
            filepath = settings.MODEL_PATH
        
        # For simplicity, we'll save the whole object (in practice, save each model separately)
        import joblib
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str = None) -> 'FraudDetectionModel':
        """Load a saved model"""
        if filepath is None:
            filepath = settings.MODEL_PATH
        
        import joblib
        return joblib.load(filepath)
    
    