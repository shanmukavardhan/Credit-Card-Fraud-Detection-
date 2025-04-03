# src/features/pca_transformer.py
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from config.settings import settings

class PCATransformer:
    def __init__(self):
        self.n_components = 28
        self.pipeline = self._build_pipeline()
        self.feature_names = None

    def _build_pipeline(self):
        # Define numeric features (matching creditcard.csv)
        numeric_features = [
            'amount',
            'merchant_latitude', 
            'merchant_longitude',
            'user_home_latitude',
            'user_home_longitude',
            'user_ip_latitude',
            'user_ip_longitude',
            'device_age_days',
            'transaction_hour',
            'merchant_risk_score',
            'merchant_avg_transaction',
            'merchant_chargeback_rate',
            'merchant_age_days',
            'user_age',
            'user_credit_score',
            'user_account_age_days',
            'user_avg_transaction',
            'user_session_duration_avg',
            'ip_distance_km',
            'transaction_duration_sec',
            'transactions_last_1h',
            'transactions_last_24h',
            'amount_to_avg_balance_ratio',
            'location_velocity_kmh',
            'composite_risk_score'
        ]
        
        # Define categorical features (matching creditcard.csv)
        categorical_features = [
            'merchant_category',
            'merchant_country',
            'device_type',
            'payment_type',
            'card_brand',
            'card_type',
            'user_ip_country',
            'shipping_country',
            'merchant_industry',
            'device_os'
        ]

        # Build preprocessing pipelines
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        # Full pipeline with PCA
        return Pipeline([
            ('preprocessor', preprocessor),
            ('pca', PCA(n_components=self.n_components))
        ])

    def fit(self, X: pd.DataFrame):
        """Fit the PCA transformer to the data"""
        self.pipeline.fit(X)
        self.feature_names = [f'V{i}' for i in range(1, self.n_components+1)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted PCA model"""
        if not hasattr(self, 'feature_names'):
            raise ValueError("Transformer not fitted. Call fit() first.")
        transformed = self.pipeline.transform(X)
        return pd.DataFrame(transformed, columns=self.feature_names)

    def save(self, path=None):
        """Save the trained transformer"""
        path = path or settings.MODEL_DIR / "pca_transformer.joblib"
        joblib.dump(self, path)

    @classmethod
    def load(cls, path=None):
        """Load a saved transformer"""
        path = path or settings.MODEL_DIR / "pca_transformer.joblib"
        return joblib.load(path)
    
    def get_pca_components(self) -> pd.DataFrame:
        """Returns PCA components (V1-V28) with original feature names"""
        if not hasattr(self, 'feature_names'):
            raise ValueError("PCA not fitted yet. Call fit() first.")
        
        # Get the PCA step from the pipeline
        pca = self.pipeline.named_steps['pca']
        
        # Get feature names from the preprocessor
        numeric_features = self.pipeline.named_steps['preprocessor'].transformers_[0][2]
        categorical_features = self.pipeline.named_steps['preprocessor'].transformers_[1][2]
        
        # Get one-hot encoded categorical feature names
        onehot = self.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder']
        categorical_onehot_names = onehot.get_feature_names_out(categorical_features)
        
        # Combine all original feature names
        original_feature_names = list(numeric_features) + list(categorical_onehot_names)
        
        # Create DataFrame showing how original features map to PCA components
        components_df = pd.DataFrame(
            pca.components_.T,  # Transpose to match sklearn's convention
            columns=self.feature_names,  # V1-V28
            index=original_feature_names  # Original features
        )
        return components_df

    def get_explained_variance(self) -> pd.DataFrame:
        """Returns explained variance ratio for each component"""
        pca = self.pipeline.named_steps['pca']
        return pd.DataFrame({
            'Component': self.feature_names,
            'Explained Variance': pca.explained_variance_ratio_
        }).sort_values('Explained Variance', ascending=False)