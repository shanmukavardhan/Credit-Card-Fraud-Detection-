import logging
from pathlib import Path
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.model import FraudDetectionModel
from config.settings import settings
import pandas as pd
# scripts/train_pca_model.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add project root to path

from src.models.pca_fraud_detector import PCAFraudDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting model training pipeline")
    
    # Create necessary directories
    settings.DATA_DIR.mkdir(exist_ok=True)
    settings.PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    settings.MODEL_DIR.mkdir(exist_ok=True)
    
    try:
        # Load data
        logger.info("Loading data")
        data_loader = DataLoader()
        df = data_loader.load_raw_data()
        
        # Preprocess data
        logger.info("Preprocessing data")
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess_data(df)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
        
        # Save processed data
        data_loader.save_processed_data(pd.concat([X_train, y_train], axis=1), "train.csv")
        data_loader.save_processed_data(pd.concat([X_val, y_val], axis=1), "val.csv")
        data_loader.save_processed_data(pd.concat([X_test, y_test], axis=1), "test.csv")
        
        # Train model
        logger.info("Training model")
        model = FraudDetectionModel()
        model.train_ensemble(X_train, y_train, X_val, y_val)
        
        # Save model
        logger.info("Saving model")
        model.save()
        
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()