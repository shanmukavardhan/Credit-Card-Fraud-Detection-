from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import logging
from src.models.pca_fraud_detector import PCAFraudDetector

router = APIRouter()
logger = logging.getLogger(__name__)

# Load model
try:
    detector = PCAFraudDetector.load()
    logger.info("PCA Fraud Detector loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    detector = None

# Expanded RawTransaction model to match your earlier data
class RawTransaction(BaseModel):
    # Transaction Details
    transaction_id: str
    timestamp: str
    merchant_id: str
    user_id: str
    amount: float
    currency: str
    
    # Geographic Features
    merchant_latitude: Optional[float] = None
    merchant_longitude: Optional[float] = None
    merchant_city: Optional[str] = None
    merchant_state: Optional[str] = None
    merchant_country: Optional[str] = None
    merchant_zip: Optional[str] = None
    user_home_latitude: Optional[float] = None
    user_home_longitude: Optional[float] = None
    user_ip_region: Optional[str] = None
    user_ip_country: Optional[str] = None
    
    # Merchant Features
    merchant_category: Optional[str] = None
    merchant_risk_score: Optional[float] = None
    merchant_avg_transaction: Optional[float] = None
    merchant_chargeback_rate: Optional[float] = None
    merchant_age_days: Optional[int] = None
    
    # User Features
    user_age: Optional[int] = None
    user_income_bracket: Optional[float] = None
    user_credit_score: Optional[int] = None
    user_account_age_days: Optional[int] = None
    user_avg_transaction: Optional[float] = None
    user_prev_chargebacks: Optional[int] = None
    
    # Device Features
    device_type: Optional[str] = None
    device_os: Optional[str] = None
    device_browser: Optional[str] = None
    device_age_days: Optional[int] = None
    device_fingerprint: Optional[str] = None
    device_velocity_kmh: Optional[float] = None
    
    # Transaction Timing
    transaction_hour: Optional[int] = None
    transaction_day_of_week: Optional[int] = None
    transaction_day_of_month: Optional[int] = None
    days_since_last_transaction: Optional[int] = None
    seconds_since_last_login: Optional[int] = None
    
    # Behavioral Features
    transactions_last_1h: Optional[int] = None
    transactions_last_24h: Optional[int] = None
    transactions_last_7d: Optional[int] = None
    avg_transaction_gap_1w_sec: Optional[int] = None
    user_typical_transaction_hour: Optional[float] = None
    
    # Payment Method
    payment_type: Optional[str] = None
    card_brand: Optional[str] = None
    card_type: Optional[str] = None
    card_expiry_months: Optional[int] = None
    card_tokenized: Optional[int] = None
    card_3ds_authenticated: Optional[int] = None
    card_issuer: Optional[str] = None
    
    # Network Features
    ip_risk_score: Optional[float] = None
    ip_proxy_used: Optional[int] = None
    ip_asn: Optional[str] = None
    ip_distance_km: Optional[float] = None
    ip_city_match: Optional[int] = None
    
    # Transaction Context
    basket_size: Optional[int] = None
    contains_digital_goods: Optional[int] = None
    contains_high_risk_items: Optional[int] = None
    same_as_shipping_address: Optional[int] = None
    shipping_speed: Optional[str] = None
    
    # User Behavior
    session_duration_sec: Optional[int] = None
    mouse_movements: Optional[int] = None
    keystroke_speed: Optional[float] = None
    login_attempts: Optional[int] = None
    page_activity_score: Optional[float] = None
    
    # Merchant-Side Data
    merchant_avs_response: Optional[str] = None
    merchant_cvv_response: Optional[str] = None
    merchant_risk_decision: Optional[str] = None
    merchant_fraud_filters_triggered: Optional[int] = None
    
    # Historical Patterns
    user_hist_chargeback_rate: Optional[float] = None
    user_hist_merchant_transactions: Optional[int] = None
    user_hist_category_transactions: Optional[int] = None
    user_hist_amount_deviation: Optional[float] = None
    
    # Derived Features
    amount_to_avg_balance_ratio: Optional[float] = None
    location_velocity_kmh: Optional[float] = None
    transaction_size_percentile: Optional[float] = None
    time_of_day_risk_score: Optional[float] = None
    device_trust_score: Optional[float] = None
    
    # Supplemental Data
    user_email_domain: Optional[str] = None
    user_phone_carrier: Optional[str] = None
    user_social_media_linked: Optional[int] = None
    user_2fa_enabled: Optional[int] = None
    user_has_verified_identity: Optional[int] = None
    
    # Merchant Technical
    merchant_checkout_version: Optional[str] = None
    merchant_platform: Optional[str] = None
    merchant_tls_version: Optional[str] = None
    merchant_pci_compliant: Optional[int] = None
    
    # Temporal Patterns
    hourly_transaction_velocity: Optional[float] = None
    weekly_pattern_match: Optional[float] = None
    holiday_indicator: Optional[int] = None
    seasonal_risk_factor: Optional[float] = None
    
    # Cross-Feature Interactions
    high_value_new_device_score: Optional[float] = None
    category_velocity_alert: Optional[int] = None
    geo_ip_mismatch_score: Optional[float] = None
    browser_os_consistency: Optional[int] = None
    
    # Engineered Features (example subset; extend to 50 if needed)
    engineered_feature_1: Optional[float] = None
    engineered_feature_2: Optional[int] = None
    engineered_feature_3: Optional[int] = None
    engineered_feature_4: Optional[float] = None
    # Add more up to engineered_feature_50 as needed

@router.post("/predict_raw")
async def predict_raw(transaction: RawTransaction):
    if not detector:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert transaction to DataFrame
        raw_df = pd.DataFrame([transaction.dict()])
        
        # Fill missing values with defaults (customize based on your model needs)
        raw_df.fillna({
            "merchant_risk_score": 0.0,
            "merchant_chargeback_rate": 0.0,
            "user_age": 30,
            "user_credit_score": 600,
            "merchant_avg_transaction": 100.0,
            "user_avg_transaction": 100.0,
            "device_velocity_kmh": 0.0,
            "ip_risk_score": 0.0,
            "transactions_last_1h": 0,
            "transactions_last_24h": 0,
            "transactions_last_7d": 0,
            "user_hist_chargeback_rate": 0.0,
            "time_of_day_risk_score": 0.0,
            "device_trust_score": 0.5,
            # Add more defaults as necessary
        }, inplace=True)
        
        # Make prediction
        pred_class, pred_proba = detector.predict(raw_df)
        
        return {
            "is_fraud": bool(pred_class[0]),
            "probability": float(pred_proba[0]),
            "threshold": 0.5  # Adjust this based on your model or settings
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))