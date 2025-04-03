import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Project
    PROJECT_NAME = "Credit Card Fraud Detection"
    PROJECT_VERSION = "1.0.0"
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODEL_DIR = DATA_DIR / "models"
    
    # Model
    MODEL_NAME = "fraud_detection_model"
    MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}.pkl"
    THRESHOLD = 0.5
    
    # API
    API_PREFIX = "/api"
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    
    # Monitoring
    MONITORING_WINDOW_SIZE = 1000
    DRIFT_THRESHOLD = 0.1
    
    # Logging
    LOGGING_CONFIG = BASE_DIR / "config" / "logging_config.py"
    

settings = Settings()