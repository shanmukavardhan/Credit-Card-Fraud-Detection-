# scripts/train_pca_model.py
import logging
import pandas as pd
import sys
from pathlib import Path

# Add the project root directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from src.models.pca_fraud_detector import PCAFraudDetector
from src.data.data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Training PCA Fraud Detector")
    
    try:
        # 1. Load raw data
        logger.info(f"Loading data from {settings.RAW_DATA_PATH}")
        raw_data = pd.read_csv(settings.RAW_DATA_PATH)
        
        # 2. Initialize and fit PCA fraud detector
        detector = PCAFraudDetector()
        logger.info("Training model...")
        detector.fit(raw_data)

        # After detector.fit()
        components = detector.pca_transformer.get_pca_components()
        variance = detector.pca_transformer.get_explained_variance()

        print("PCA Components (V1-V28):")
        print(components.head())  # Shows how original features map to V1-V28

        print("\nExplained Variance:")
        print(variance)  # Shows how much each V component contributes
        
        # 3. Save the trained model
        detector.save()
        logger.info(f"Model saved to {settings.MODEL_DIR}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()