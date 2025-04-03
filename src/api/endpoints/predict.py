from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import logging
from typing import Optional
from config.settings import settings
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize detector without loading models immediately
detector = None

def get_detector():
    global detector
    if detector is None:
        try:
            from src.models.pca_fraud_detector import PCAFraudDetector
            detector = PCAFraudDetector()
            detector.load(settings.MODEL_PATH)  # Ensure MODEL_PATH is correctly set in your settings
        except Exception as e:
            raise RuntimeError(f"Failed to load the detector: {str(e)}")
    return detector

class RawTransaction(BaseModel):
    # Core Transaction Fields
    transaction_id: str = Field(..., example="T10001")
    timestamp: str = Field(..., example="2023-07-15 14:30:22")
    merchant_id: str = Field(..., example="MERC_22861")
    user_id: str = Field(..., example="USER_8842")
    amount: float = Field(..., example=149.99)
    currency: str = Field(..., example="USD")
    transaction_channel: str = Field(..., example="web")
    transaction_source: str = Field(..., example="api")
    
    # Geographic Features
    merchant_latitude: float = Field(..., example=40.7128)
    merchant_longitude: float = Field(..., example=-74.0060)
    merchant_city: str = Field(..., example="New York")
    merchant_state: str = Field(..., example="NY")
    merchant_country: str = Field(..., example="US")
    merchant_zip: str = Field(..., example="10001")
    merchant_timezone: str = Field(..., example="America/New_York")
    user_home_latitude: float = Field(..., example=40.7306)
    user_home_longitude: float = Field(..., example=-73.9352)
    user_ip_latitude: float = Field(..., example=40.7128)
    user_ip_longitude: float = Field(..., example=-74.0060)
    user_ip_region: str = Field(..., example="NY")
    user_ip_country: str = Field(..., example="US")
    user_ip_city: str = Field(..., example="New York")
    shipping_latitude: Optional[float] = Field(None, example=40.7128)
    shipping_longitude: Optional[float] = Field(None, example=-74.0060)
    shipping_country: Optional[str] = Field(None, example="US")
    
    # Merchant Features
    merchant_category: str = Field(..., example="electronics")
    merchant_risk_score: float = Field(..., example=0.12)
    merchant_avg_transaction: float = Field(..., example=89.50)
    merchant_chargeback_rate: float = Field(..., example=0.008)
    merchant_age_days: int = Field(..., example=720)
    merchant_volume_tier: str = Field(..., example="high")
    merchant_processing_history: int = Field(..., example=12)
    merchant_industry: str = Field(..., example="ecommerce")
    merchant_business_type: str = Field(..., example="b2c")
    
    # User Features
    user_age: int = Field(..., example=34)
    user_income_bracket: int = Field(..., example=75000)
    user_credit_score: int = Field(..., example=720)
    user_account_age_days: int = Field(..., example=1095)
    user_avg_transaction: float = Field(..., example=112.30)
    user_prev_chargebacks: int = Field(..., example=0)
    user_fraud_reports: int = Field(..., example=0)
    user_login_frequency: float = Field(..., example=2.5)
    user_session_duration_avg: float = Field(..., example=120.5)
    user_device_count: int = Field(..., example=3)
    
    # Device Features
    device_type: str = Field(..., example="mobile")
    device_os: str = Field(..., example="iOS 15.4")
    device_browser: str = Field(..., example="Safari")
    device_age_days: int = Field(..., example=180)
    device_fingerprint: str = Field(..., example="AA3829DF29")
    device_velocity_kmh: float = Field(..., example=0.0)
    device_screen_resolution: str = Field(..., example="1125x2436")
    device_language: str = Field(..., example="en-US")
    device_timezone: str = Field(..., example="America/New_York")
    device_fonts_hash: str = Field(..., example="a1b2c3d4")
    
    # Transaction Timing
    transaction_hour: int = Field(..., example=14)
    transaction_day_of_week: int = Field(..., example=2)  # Tuesday
    transaction_day_of_month: int = Field(..., example=15)
    transaction_day_of_year: int = Field(..., example=196)
    days_since_last_transaction: int = Field(..., example=2)
    seconds_since_last_login: int = Field(..., example=342)
    transaction_duration_sec: float = Field(..., example=45.2)
    
    # Behavioral Features
    transactions_last_1h: int = Field(..., example=0)
    transactions_last_24h: int = Field(..., example=3)
    transactions_last_7d: int = Field(..., example=12)
    transactions_last_30d: int = Field(..., example=42)
    avg_transaction_gap_1w_sec: float = Field(..., example=86400)
    user_typical_transaction_hour: float = Field(..., example=13.5)
    user_typical_transaction_amount: float = Field(..., example=112.30)
    user_typical_transaction_location: str = Field(..., example="home")
    
    # Payment Method
    payment_type: str = Field(..., example="credit_card")
    card_brand: str = Field(..., example="VISA")
    card_type: str = Field(..., example="platinum")
    card_expiry_months: int = Field(..., example=24)
    card_tokenized: int = Field(..., example=1)
    card_3ds_authenticated: int = Field(..., example=1)
    card_issuer: str = Field(..., example="Chase")
    card_issuer_country: str = Field(..., example="US")
    payment_processor: str = Field(..., example="stripe")
    payment_authentication_method: str = Field(..., example="3ds")
    
    # Network Features
    ip_risk_score: float = Field(..., example=0.05)
    ip_proxy_used: int = Field(..., example=0)
    ip_asn: str = Field(..., example="AS7922")
    ip_distance_km: float = Field(..., example=15.2)
    ip_city_match: int = Field(..., example=1)
    ip_country_match: int = Field(..., example=1)
    ip_isp: str = Field(..., example="Comcast")
    ip_connection_type: str = Field(..., example="broadband")
    ip_anonymous_proxy: int = Field(..., example=0)
    ip_reputation_score: float = Field(..., example=0.95)
    
    # Transaction Context
    basket_size: int = Field(..., example=1)
    contains_digital_goods: int = Field(..., example=0)
    contains_high_risk_items: int = Field(..., example=0)
    same_as_shipping_address: int = Field(..., example=1)
    shipping_speed: str = Field(..., example="standard")
    order_amount_usd: float = Field(..., example=149.99)
    order_currency: str = Field(..., example="USD")
    order_discount_amount: float = Field(..., example=0.0)
    order_has_coupon: int = Field(..., example=0)
    order_item_count: int = Field(..., example=1)
    
    # User Behavior
    session_duration_sec: int = Field(..., example=142)
    mouse_movements: int = Field(..., example=28)
    keystroke_speed: float = Field(..., example=4.2)
    login_attempts: int = Field(..., example=1)
    page_activity_score: float = Field(..., example=0.88)
    form_fill_time: float = Field(..., example=12.5)
    click_pattern_score: float = Field(..., example=0.92)
    mouse_velocity: float = Field(..., example=1.2)
    scroll_behavior: str = Field(..., example="steady")
    page_transitions: int = Field(..., example=3)
    
    # Merchant-side Data
    merchant_avs_response: str = Field(..., example="Y")
    merchant_cvv_response: str = Field(..., example="M")
    merchant_risk_decision: str = Field(..., example="accept")
    merchant_fraud_filters_triggered: int = Field(..., example=0)
    merchant_decision_time_ms: int = Field(..., example=120)
    merchant_score: float = Field(..., example=0.85)
    merchant_velocity_check: int = Field(..., example=1)
    merchant_blacklist_check: int = Field(..., example=0)
    merchant_whitelist_check: int = Field(..., example=1)
    merchant_custom_rules_triggered: int = Field(..., example=0)
    
    # Historical Patterns
    user_hist_chargeback_rate: float = Field(..., example=0.0)
    user_hist_merchant_transactions: int = Field(..., example=8)
    user_hist_category_transactions: int = Field(..., example=42)
    user_hist_amount_deviation: float = Field(..., example=0.32)
    user_hist_time_deviation: float = Field(..., example=0.45)
    user_hist_location_deviation: float = Field(..., example=0.12)
    user_hist_device_consistency: float = Field(..., example=0.95)
    user_hist_payment_consistency: float = Field(..., example=0.98)
    user_hist_behavior_score: float = Field(..., example=0.92)
    user_hist_risk_score: float = Field(..., example=0.05)
    
    # Derived Features
    amount_to_avg_balance_ratio: float = Field(..., example=0.012)
    location_velocity_kmh: float = Field(..., example=0.0)
    transaction_size_percentile: float = Field(..., example=0.65)
    time_of_day_risk_score: float = Field(..., example=0.08)
    device_trust_score: float = Field(..., example=0.92)
    behavioral_anomaly_score: float = Field(..., example=0.05)
    network_risk_score: float = Field(..., example=0.03)
    payment_anomaly_score: float = Field(..., example=0.01)
    temporal_anomaly_score: float = Field(..., example=0.07)
    composite_risk_score: float = Field(..., example=0.12)
    
    # Supplemental Data
    user_email_domain: str = Field(..., example="gmail.com")
    user_phone_carrier: str = Field(..., example="Verizon")
    user_social_media_linked: int = Field(..., example=1)
    user_2fa_enabled: int = Field(..., example=1)
    user_has_verified_identity: int = Field(..., example=1)
    user_email_age_days: int = Field(..., example=1825)
    user_phone_age_days: int = Field(..., example=730)
    user_account_verified: int = Field(..., example=1)
    user_kyc_level: int = Field(..., example=2)
    user_reputation_score: float = Field(..., example=0.95)
    
    # Merchant Technical
    merchant_checkout_version: str = Field(..., example="v3.2")
    merchant_platform: str = Field(..., example="shopify")
    merchant_tls_version: str = Field(..., example="1.3")
    merchant_pci_compliant: int = Field(..., example=1)
    merchant_fraud_tools: str = Field(..., example="kount,sift")
    merchant_authentication_methods: str = Field(..., example="3ds,biometric")
    merchant_checkout_flow: str = Field(..., example="one_page")
    merchant_anti_fraud_system: str = Field(..., example="internal")
    merchant_decision_engine: str = Field(..., example="rules_v3")
    merchant_data_quality_score: float = Field(..., example=0.98)
    
    # Temporal Patterns
    hourly_transaction_velocity: float = Field(..., example=0.4)
    weekly_pattern_match: float = Field(..., example=0.87)
    holiday_indicator: int = Field(..., example=0)
    seasonal_risk_factor: float = Field(..., example=0.05)
    time_since_last_chargeback: Optional[float] = Field(None, example=90.5)
    time_since_last_fraud: Optional[float] = Field(None, example=180.2)
    transaction_time_deviation: float = Field(..., example=0.32)
    user_activity_time_score: float = Field(..., example=0.88)
    merchant_peak_hour: int = Field(..., example=1)
    time_based_risk_score: float = Field(..., example=0.12)
    
    # Cross-feature Interactions
    high_value_new_device_score: float = Field(..., example=0.03)
    category_velocity_alert: int = Field(..., example=0)
    geo_ip_mismatch_score: float = Field(..., example=0.01)
    browser_os_consistency: int = Field(..., example=1)
    device_payment_mismatch: int = Field(..., example=0)
    shipping_billing_discrepancy: int = Field(..., example=0)
    behavioral_payment_anomaly: float = Field(..., example=0.05)
    temporal_location_risk: float = Field(..., example=0.08)
    user_merchant_trust_score: float = Field(..., example=0.92)
    composite_interaction_score: float = Field(..., example=0.15)

    # Engineered Features (50 total)
    engineered_feature_1: Optional[float] = Field(None, example=0.5)
    engineered_feature_2: Optional[float] = Field(None, example=0.3)
    engineered_feature_3: Optional[float] = Field(None, example=0.7)
    engineered_feature_4: Optional[float] = Field(None, example=0.2)
    engineered_feature_5: Optional[float] = Field(None, example=0.9)
    engineered_feature_6: Optional[float] = Field(None, example=0.1)
    engineered_feature_7: Optional[float] = Field(None, example=0.6)
    engineered_feature_8: Optional[float] = Field(None, example=0.4)
    engineered_feature_9: Optional[float] = Field(None, example=0.8)
    engineered_feature_10: Optional[float] = Field(None, example=0.25)
    engineered_feature_11: Optional[float] = Field(None, example=0.75)
    engineered_feature_12: Optional[float] = Field(None, example=0.35)
    engineered_feature_13: Optional[float] = Field(None, example=0.65)
    engineered_feature_14: Optional[float] = Field(None, example=0.15)
    engineered_feature_15: Optional[float] = Field(None, example=0.85)
    engineered_feature_16: Optional[float] = Field(None, example=0.45)
    engineered_feature_17: Optional[float] = Field(None, example=0.55)
    engineered_feature_18: Optional[float] = Field(None, example=0.05)
    engineered_feature_19: Optional[float] = Field(None, example=0.95)
    engineered_feature_20: Optional[float] = Field(None, example=0.28)
    engineered_feature_21: Optional[float] = Field(None, example=0.72)
    engineered_feature_22: Optional[float] = Field(None, example=0.38)
    engineered_feature_23: Optional[float] = Field(None, example=0.62)
    engineered_feature_24: Optional[float] = Field(None, example=0.18)
    engineered_feature_25: Optional[float] = Field(None, example=0.82)
    engineered_feature_26: Optional[float] = Field(None, example=0.42)
    engineered_feature_27: Optional[float] = Field(None, example=0.58)
    engineered_feature_28: Optional[float] = Field(None, example=0.08)
    engineered_feature_29: Optional[float] = Field(None, example=0.92)
    engineered_feature_30: Optional[float] = Field(None, example=0.22)
    engineered_feature_31: Optional[float] = Field(None, example=0.78)
    engineered_feature_32: Optional[float] = Field(None, example=0.32)
    engineered_feature_33: Optional[float] = Field(None, example=0.68)
    engineered_feature_34: Optional[float] = Field(None, example=0.12)
    engineered_feature_35: Optional[float] = Field(None, example=0.88)
    engineered_feature_36: Optional[float] = Field(None, example=0.48)
    engineered_feature_37: Optional[float] = Field(None, example=0.52)
    engineered_feature_38: Optional[float] = Field(None, example=0.02)
    engineered_feature_39: Optional[float] = Field(None, example=0.98)
    engineered_feature_40: Optional[float] = Field(None, example=0.24)
    engineered_feature_41: Optional[float] = Field(None, example=0.76)
    engineered_feature_42: Optional[float] = Field(None, example=0.34)
    engineered_feature_43: Optional[float] = Field(None, example=0.66)
    engineered_feature_44: Optional[float] = Field(None, example=0.14)
    engineered_feature_45: Optional[float] = Field(None, example=0.86)
    engineered_feature_46: Optional[float] = Field(None, example=0.46)
    engineered_feature_47: Optional[float] = Field(None, example=0.54)
    engineered_feature_48: Optional[float] = Field(None, example=0.04)
    engineered_feature_49: Optional[float] = Field(None, example=0.96)
    engineered_feature_50: Optional[float] = Field(None, example=0.26)

    class Config:
        schema_extra = {
            "example": {
                # Core Transaction Fields
                "transaction_id": "T10001",
                "timestamp": "2023-07-15 14:30:22",
                "merchant_id": "MERC_22861",
                "user_id": "USER_8842",
                "amount": 149.99,
                "currency": "USD",
                "transaction_channel": "web",
                "transaction_source": "api",
                
                # Geographic Features
                "merchant_latitude": 40.7128,
                "merchant_longitude": -74.0060,
                "merchant_city": "New York",
                "merchant_state": "NY",
                "merchant_country": "US",
                "merchant_zip": "10001",
                "merchant_timezone": "America/New_York",
                "user_home_latitude": 40.7306,
                "user_home_longitude": -73.9352,
                "user_ip_latitude": 40.7128,
                "user_ip_longitude": -74.0060,
                "user_ip_region": "NY",
                "user_ip_country": "US",
                "user_ip_city": "New York",
                "shipping_latitude": 40.7128,
                "shipping_longitude": -74.0060,
                "shipping_country": "US",
                
                # Merchant Features
                "merchant_category": "electronics",
                "merchant_risk_score": 0.12,
                "merchant_avg_transaction": 89.50,
                "merchant_chargeback_rate": 0.008,
                "merchant_age_days": 720,
                "merchant_volume_tier": "high",
                "merchant_processing_history": 12,
                "merchant_industry": "ecommerce",
                "merchant_business_type": "b2c",
                
                # User Features
                "user_age": 34,
                "user_income_bracket": 75000,
                "user_credit_score": 720,
                "user_account_age_days": 1095,
                "user_avg_transaction": 112.30,
                "user_prev_chargebacks": 0,
                "user_fraud_reports": 0,
                "user_login_frequency": 2.5,
                "user_session_duration_avg": 120.5,
                "user_device_count": 3,
                
                # Device Features
                "device_type": "mobile",
                "device_os": "iOS 15.4",
                "device_browser": "Safari",
                "device_age_days": 180,
                "device_fingerprint": "AA3829DF29",
                "device_velocity_kmh": 0.0,
                "device_screen_resolution": "1125x2436",
                "device_language": "en-US",
                "device_timezone": "America/New_York",
                "device_fonts_hash": "a1b2c3d4",
                
                # Transaction Timing
                "transaction_hour": 14,
                "transaction_day_of_week": 2,
                "transaction_day_of_month": 15,
                "transaction_day_of_year": 196,
                "days_since_last_transaction": 2,
                "seconds_since_last_login": 342,
                "transaction_duration_sec": 45.2,
                
                # Behavioral Features
                "transactions_last_1h": 0,
                "transactions_last_24h": 3,
                "transactions_last_7d": 12,
                "transactions_last_30d": 42,
                "avg_transaction_gap_1w_sec": 86400,
                "user_typical_transaction_hour": 13.5,
                "user_typical_transaction_amount": 112.30,
                "user_typical_transaction_location": "home",
                
                # Payment Method
                "payment_type": "credit_card",
                "card_brand": "VISA",
                "card_type": "platinum",
                "card_expiry_months": 24,
                "card_tokenized": 1,
                "card_3ds_authenticated": 1,
                "card_issuer": "Chase",
                "card_issuer_country": "US",
                "payment_processor": "stripe",
                "payment_authentication_method": "3ds",
                
                # Network Features
                "ip_risk_score": 0.05,
                "ip_proxy_used": 0,
                "ip_asn": "AS7922",
                "ip_distance_km": 15.2,
                "ip_city_match": 1,
                "ip_country_match": 1,
                "ip_isp": "Comcast",
                "ip_connection_type": "broadband",
                "ip_anonymous_proxy": 0,
                "ip_reputation_score": 0.95,
                
                # Transaction Context
                "basket_size": 1,
                "contains_digital_goods": 0,
                "contains_high_risk_items": 0,
                "same_as_shipping_address": 1,
                "shipping_speed": "standard",
                "order_amount_usd": 149.99,
                "order_currency": "USD",
                "order_discount_amount": 0.0,
                "order_has_coupon": 0,
                "order_item_count": 1,
                
                # User Behavior
                "session_duration_sec": 142,
                "mouse_movements": 28,
                "keystroke_speed": 4.2,
                "login_attempts": 1,
                "page_activity_score": 0.88,
                "form_fill_time": 12.5,
                "click_pattern_score": 0.92,
                "mouse_velocity": 1.2,
                "scroll_behavior": "steady",
                "page_transitions": 3,
                
                # Merchant-side Data
                "merchant_avs_response": "Y",
                "merchant_cvv_response": "M",
                "merchant_risk_decision": "accept",
                "merchant_fraud_filters_triggered": 0,
                "merchant_decision_time_ms": 120,
                "merchant_score": 0.85,
                "merchant_velocity_check": 1,
                "merchant_blacklist_check": 0,
                "merchant_whitelist_check": 1,
                "merchant_custom_rules_triggered": 0,
                
                # Historical Patterns
                "user_hist_chargeback_rate": 0.0,
                "user_hist_merchant_transactions": 8,
                "user_hist_category_transactions": 42,
                "user_hist_amount_deviation": 0.32,
                "user_hist_time_deviation": 0.45,
                "user_hist_location_deviation": 0.12,
                "user_hist_device_consistency": 0.95,
                "user_hist_payment_consistency": 0.98,
                "user_hist_behavior_score": 0.92,
                "user_hist_risk_score": 0.05,
                
                # Derived Features
                "amount_to_avg_balance_ratio": 0.012,
                "location_velocity_kmh": 0.0,
                "transaction_size_percentile": 0.65,
                "time_of_day_risk_score": 0.08,
                "device_trust_score": 0.92,
                "behavioral_anomaly_score": 0.05,
                "network_risk_score": 0.03,
                "payment_anomaly_score": 0.01,
                "temporal_anomaly_score": 0.07,
                "composite_risk_score": 0.12,
                
                # Supplemental Data
                "user_email_domain": "gmail.com",
                "user_phone_carrier": "Verizon",
                "user_social_media_linked": 1,
                "user_2fa_enabled": 1,
                "user_has_verified_identity": 1,
                "user_email_age_days": 1825,
                "user_phone_age_days": 730,
                "user_account_verified": 1,
                "user_kyc_level": 2,
                "user_reputation_score": 0.95,
                
                # Merchant Technical
                "merchant_checkout_version": "v3.2",
                "merchant_platform": "shopify",
                "merchant_tls_version": "1.3",
                "merchant_pci_compliant": 1,
                "merchant_fraud_tools": "kount,sift",
                "merchant_authentication_methods": "3ds,biometric",
                "merchant_checkout_flow": "one_page",
                "merchant_anti_fraud_system": "internal",
                "merchant_decision_engine": "rules_v3",
                "merchant_data_quality_score": 0.98,
                
                # Temporal Patterns
                "hourly_transaction_velocity": 0.4,
                "weekly_pattern_match": 0.87,
                "holiday_indicator": 0,
                "seasonal_risk_factor": 0.05,
                "time_since_last_chargeback": 90.5,
                "time_since_last_fraud": 180.2,
                "transaction_time_deviation": 0.32,
                "user_activity_time_score": 0.88,
                "merchant_peak_hour": 1,
                "time_based_risk_score": 0.12,
                
                # Cross-feature Interactions
                "high_value_new_device_score": 0.03,
                "category_velocity_alert": 0,
                "geo_ip_mismatch_score": 0.01,
                "browser_os_consistency": 1,
                "device_payment_mismatch": 0,
                "shipping_billing_discrepancy": 0,
                "behavioral_payment_anomaly": 0.05,
                "temporal_location_risk": 0.08,
                "user_merchant_trust_score": 0.92,
                "composite_interaction_score": 0.15,

                # Engineered Features (50 total)
                "engineered_feature_1": 0.5,
                "engineered_feature_2": 0.3,
                "engineered_feature_3": 0.7,
                "engineered_feature_4": 0.2,
                "engineered_feature_5": 0.9,
                "engineered_feature_6": 0.1,
                "engineered_feature_7": 0.6,
                "engineered_feature_8": 0.4,
                "engineered_feature_9": 0.8,
                "engineered_feature_10": 0.25,
                "engineered_feature_11": 0.75,
                "engineered_feature_12": 0.35,
                "engineered_feature_13": 0.65,
                "engineered_feature_14": 0.15,
                "engineered_feature_15": 0.85,
                "engineered_feature_16": 0.45,
                "engineered_feature_17": 0.55,
                "engineered_feature_18": 0.05,
                "engineered_feature_19": 0.95,
                "engineered_feature_20": 0.28,
                "engineered_feature_21": 0.72,
                "engineered_feature_22": 0.38,
                "engineered_feature_23": 0.62,
                "engineered_feature_24": 0.18,
                "engineered_feature_25": 0.82,
                "engineered_feature_26": 0.42,
                "engineered_feature_27": 0.58,
                "engineered_feature_28": 0.08,
                "engineered_feature_29": 0.92,
                "engineered_feature_30": 0.22,
                "engineered_feature_31": 0.78,
                "engineered_feature_32": 0.32,
                "engineered_feature_33": 0.68,
                "engineered_feature_34": 0.12,
                "engineered_feature_35": 0.88,
                "engineered_feature_36": 0.48,
                "engineered_feature_37": 0.52,
                "engineered_feature_38": 0.02,
                "engineered_feature_39": 0.98,
                "engineered_feature_40": 0.24,
                "engineered_feature_41": 0.76,
                "engineered_feature_42": 0.34,
                "engineered_feature_43": 0.66,
                "engineered_feature_44": 0.14,
                "engineered_feature_45": 0.86,
                "engineered_feature_46": 0.46,
                "engineered_feature_47": 0.54,
                "engineered_feature_48": 0.04,
                "engineered_feature_49": 0.96,
                "engineered_feature_50": 0.26
            }
        }

@router.post("/predict_raw")
async def predict_raw(transaction: RawTransaction):
    try:
        detector = get_detector()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {str(e)}")
    
    try:
        # Convert to DataFrame and ensure proper data types
        transaction_data = transaction.dict()
        raw_df = pd.DataFrame([transaction_data])
        
        # Ensure numeric fields are properly converted
        numeric_cols = [col for col in raw_df.columns if raw_df[col].dtype in ['object', 'bool']]
        for col in numeric_cols:
            try:
                raw_df[col] = pd.to_numeric(raw_df[col])
            except (ValueError, TypeError):
                pass  # Keep as-is if conversion fails
                
        # Handle missing values for numeric columns
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns
        raw_df[numeric_cols] = raw_df[numeric_cols].fillna(0)
        
        # Make prediction
        pred_class, pred_proba = detector.predict(raw_df)
        
        return {
            "is_fraud": bool(pred_class[0]),
            "probability": float(pred_proba[0]),
            "threshold": settings.THRESHOLD,
            "features_used": detector.pca_transformer.feature_names
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=422,
            detail={
                "error": str(e),
                "message": "Data processing failed. Please check your input data.",
                "expected_schema": RawTransaction.schema()
            }
        )