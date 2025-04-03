from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
import logging
import logging.config
from pathlib import Path
from config.settings import settings
from src.models.model import FraudDetectionModel
from src.api.endpoints import predict_router, predict_raw_router, health_router

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for credit card fraud detection",
    version=settings.PROJECT_VERSION
)

# Setup templates
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Configure logging
try:
    logging.config.fileConfig(settings.LOGGING_CONFIG, disable_existing_loggers=False)
except Exception as e:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.warning(f"Using basic logging config. Could not load logging config: {str(e)}")

logger = logging.getLogger(__name__)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize router
router = APIRouter()

# Load model
try:
    model = FraudDetectionModel.load()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Failed to load model")

class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class PredictionResult(BaseModel):
    is_fraud: bool
    probability: float
    threshold: float

# Make sure these routers are properly configured
app.include_router(predict_router, prefix="/api")
app.include_router(predict_raw_router, prefix="/api")  # This is used in your frontend
app.include_router(health_router, prefix="/api")

@app.get("/transaction", response_class=HTMLResponse)
async def transaction_form(request: Request):
    return templates.TemplateResponse("transaction.html", {"request": request})

@router.post("/predict", response_model=PredictionResult)
async def predict(transaction: Transaction):
    """Make prediction using V1-V28 features"""
    try:
        # Convert to DataFrame with correct column order
        transaction_dict = transaction.dict()
        cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        df = pd.DataFrame([transaction_dict])[cols]
        
        # Make prediction
        pred_class, pred_proba = model.predict(df.values)
        
        return {
            "is_fraud": bool(pred_class[0]),
            "probability": float(pred_proba[0]),
            "threshold": settings.THRESHOLD
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class RawTransactionInput(BaseModel):
    # Transaction Details
    amount: float
    time: float
    currency: str
    transaction_channel: str
    
    # Geographic Information
    merchant_latitude: Optional[float] = None
    merchant_longitude: Optional[float] = None
    merchant_city: Optional[str] = None
    merchant_country: Optional[str] = None
    user_ip_latitude: Optional[float] = None
    user_ip_longitude: Optional[float] = None
    ip_distance_km: Optional[float] = None
    
    # Merchant Information
    merchant_category: Optional[str] = None
    merchant_risk_score: Optional[float] = None
    merchant_chargeback_rate: Optional[float] = None
    
    # User Information
    user_age: Optional[int] = None
    user_credit_score: Optional[int] = None
    user_prev_chargebacks: Optional[int] = None
    
    # Device Information
    device_type: Optional[str] = None
    device_os: Optional[str] = None
    device_browser: Optional[str] = None
    
    # Payment Information
    payment_type: Optional[str] = None
    card_brand: Optional[str] = None
    card_3ds_authenticated: Optional[int] = None
    
    # Derived Features
    transaction_hour: Optional[int] = None
    transaction_day_of_week: Optional[int] = None
    transaction_day_of_month: Optional[int] = None
    transactions_last_24h: Optional[int] = None
    transactions_last_7d: Optional[int] = None
    time_of_day_risk_score: Optional[float] = None
    device_trust_score: Optional[float] = None
    behavioral_anomaly_score: Optional[float] = None

@router.post("/predict_raw", response_model=PredictionResult)
async def predict_raw(raw_input: RawTransactionInput):
    try:
        logger.info(f"Received data: {raw_input.dict()}")
        raw_data = pd.DataFrame([raw_input.dict()])
        
        # Fill missing values with defaults
        raw_data.fillna({
            "merchant_risk_score": 0.0,
            "merchant_chargeback_rate": 0.0,
            "user_age": 30,
            "user_credit_score": 600,
            # Add defaults for other fields as needed
        }, inplace=True)
        
        # Preprocess and predict
        processed_features = model.preprocess(raw_data)
        pred_class, pred_proba = model.predict(processed_features)
        
        return {
            "is_fraud": bool(pred_class[0]),
            "probability": float(pred_proba[0]),
            "threshold": settings.THRESHOLD,
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router
app.include_router(router, prefix="/api")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/check-fraud", response_class=HTMLResponse)
async def check_fraud(
    request: Request,
    amount: float = Form(...),
    time: float = Form(...),
    v1: float = Form(...),
    v2: float = Form(...),
    v3: float = Form(...),
    v4: float = Form(...),
    v5: float = Form(...),
    v6: float = Form(...),
    v7: float = Form(...),
    v8: float = Form(...),
    v9: float = Form(...),
    v10: float = Form(...),
    v11: float = Form(...),
    v12: float = Form(...),
    v13: float = Form(...),
    v14: float = Form(...),
    v15: float = Form(...),
    v16: float = Form(...),
    v17: float = Form(...),
    v18: float = Form(...),
    v19: float = Form(...),
    v20: float = Form(...),
    v21: float = Form(...),
    v22: float = Form(...),
    v23: float = Form(...),
    v24: float = Form(...),
    v25: float = Form(...),
    v26: float = Form(...),
    v27: float = Form(...),
    v28: float = Form(...)
):
    # Convert form data to API format
    transaction = {
        "Amount": amount,
        "Time": time,
        "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
        "V6": v6, "V7": v7, "V8": v8, "V9": v9, "V10": v10,
        "V11": v11, "V12": v12, "V13": v13, "V14": v14, "V15": v15,
        "V16": v16, "V17": v17, "V18": v18, "V19": v19, "V20": v20,
        "V21": v21, "V22": v22, "V23": v23, "V24": v24, "V25": v25,
        "V26": v26, "V27": v27, "V28": v28
    }
    
    # Call your existing API
    response = await predict(Transaction(**transaction))
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "is_fraud": response["is_fraud"],
        "probability": response["probability"]
    })

@app.get("/api")
async def root():
    return {"message": "Credit Card Fraud Detection API"}

@app.get("/api/health")
async def health_check():
    try:
        # Simple check to see if model is loaded
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Test prediction with dummy data
        dummy_data = np.zeros((1, 30))
        _, _ = model.predict(dummy_data)
        
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")