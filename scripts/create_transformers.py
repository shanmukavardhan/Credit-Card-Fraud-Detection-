# scripts/create_transformers.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config.settings import settings
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
import logging

# scripts/train_pca_model.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add project root to path

from src.models.pca_fraud_detector import PCAFraudDetector
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Creating feature engineering transformers")
    
    # Ensure model directory exists
    settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load and preprocess data
        data_loader = DataLoader()
        preprocessor = DataPreprocessor()
        
        df = data_loader.load_raw_data()
        X, y = preprocessor.preprocess_data(df)
        
        # Create and save scaler
        scaler = StandardScaler().fit(X)
        joblib.dump(scaler, settings.SCALER_PATH)
        logger.info(f"Scaler saved to {settings.SCALER_PATH}")
        
        # Create and save PCA
        pca = PCA(n_components=28).fit(scaler.transform(X))
        joblib.dump(pca, settings.PCA_PATH)
        logger.info(f"PCA saved to {settings.PCA_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to create transformers: {str(e)}")
        raise

if __name__ == "__main__":
    main()