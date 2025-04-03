import pandas as pd
from pathlib import Path
from config.settings import settings
import numpy as np
from datetime import datetime, timedelta
import random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add project root to path

from src.models.pca_fraud_detector import PCAFraudDetector

def generate_sample_data(num_samples=1000, fraud_ratio=0.01):
    """Generate comprehensive sample transaction data"""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate timestamps
    now = datetime.now()
    timestamps = [now - timedelta(minutes=x*10) for x in range(num_samples)]
    
    # Generate core transaction data
    data = {
        # Core Transaction Fields
        "transaction_id": [f"T{10000+i}" for i in range(num_samples)],
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
        "merchant_id": [f"MERC_{random.randint(1000,9999)}" for _ in range(num_samples)],
        "user_id": [f"USER_{random.randint(1000,9999)}" for _ in range(num_samples)],
        "amount": np.round(np.random.lognormal(mean=4.5, sigma=0.8, size=num_samples), 2),
        "currency": ["USD"] * num_samples,
        "transaction_channel": random.choices(["web", "mobile", "api"], weights=[0.6, 0.3, 0.1], k=num_samples),
        "transaction_source": random.choices(["ecommerce", "pos", "atm"], weights=[0.7, 0.2, 0.1], k=num_samples),
        
        # Geographic Features
        "merchant_latitude": np.round(np.random.uniform(low=32.0, high=48.0, size=num_samples), 4),
        "merchant_longitude": np.round(np.random.uniform(low=-125.0, high=-65.0, size=num_samples), 4),
        "merchant_city": random.choices(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], k=num_samples),
        "merchant_state": random.choices(["NY", "CA", "IL", "TX", "AZ"], k=num_samples),
        "merchant_country": ["US"] * num_samples,
        "merchant_zip": [f"{random.randint(10000,99999)}" for _ in range(num_samples)],
        "merchant_timezone": random.choices(["America/New_York", "America/Los_Angeles", "America/Chicago"], k=num_samples),
        "user_home_latitude": np.round(np.random.uniform(low=32.0, high=48.0, size=num_samples), 4),
        "user_home_longitude": np.round(np.random.uniform(low=-125.0, high=-65.0, size=num_samples), 4),
        "user_ip_latitude": np.round(np.random.uniform(low=32.0, high=48.0, size=num_samples), 4),
        "user_ip_longitude": np.round(np.random.uniform(low=-125.0, high=-65.0, size=num_samples), 4),
        "user_ip_region": random.choices(["NY", "CA", "IL", "TX", "AZ"], k=num_samples),
        "user_ip_country": ["US"] * num_samples,
        "user_ip_city": random.choices(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], k=num_samples),
        "shipping_latitude": np.round(np.random.uniform(low=32.0, high=48.0, size=num_samples), 4),
        "shipping_longitude": np.round(np.random.uniform(low=-125.0, high=-65.0, size=num_samples), 4),
        "shipping_country": ["US"] * num_samples,
        
        # Merchant Features
        "merchant_category": random.choices(["electronics", "retail", "food", "travel", "services"], k=num_samples),
        "merchant_risk_score": np.round(np.random.beta(a=1, b=9, size=num_samples), 4),
        "merchant_avg_transaction": np.round(np.random.lognormal(mean=4.0, sigma=0.5, size=num_samples), 2),
        "merchant_chargeback_rate": np.round(np.random.beta(a=1, b=99, size=num_samples), 4),
        "merchant_age_days": np.random.randint(30, 365*3, size=num_samples),
        "merchant_volume_tier": random.choices(["low", "medium", "high"], weights=[0.6, 0.3, 0.1], k=num_samples),
        "merchant_processing_history": np.random.randint(0, 50, size=num_samples),
        "merchant_industry": random.choices(["ecommerce", "retail", "hospitality", "financial"], k=num_samples),
        "merchant_business_type": random.choices(["b2c", "b2b"], weights=[0.8, 0.2], k=num_samples),
        
        # User Features
        "user_age": np.random.randint(18, 80, size=num_samples),
        "user_income_bracket": np.random.choice([25000, 50000, 75000, 100000], size=num_samples),
        "user_credit_score": np.random.randint(300, 850, size=num_samples),
        "user_account_age_days": np.random.randint(1, 365*5, size=num_samples),
        "user_avg_transaction": np.round(np.random.lognormal(mean=4.0, sigma=0.5, size=num_samples), 2),
        "user_prev_chargebacks": np.random.binomial(n=10, p=0.05, size=num_samples),
        "user_fraud_reports": np.random.binomial(n=5, p=0.01, size=num_samples),
        "user_login_frequency": np.round(np.random.exponential(scale=2.0, size=num_samples), 1),
        "user_session_duration_avg": np.round(np.random.normal(loc=120, scale=30, size=num_samples), 1),
        "user_device_count": np.random.randint(1, 5, size=num_samples),
        
        # Device Features
        "device_type": random.choices(["mobile", "desktop", "tablet"], weights=[0.6, 0.3, 0.1], k=num_samples),
        "device_os": random.choices(["iOS 15.4", "Android 12", "Windows 10", "Mac OS"], k=num_samples),
        "device_browser": random.choices(["Chrome", "Safari", "Firefox", "Edge"], k=num_samples),
        "device_age_days": np.random.randint(1, 365, size=num_samples),
        "device_fingerprint": [f"DEV_{random.randint(10000,99999)}" for _ in range(num_samples)],
        "device_velocity_kmh": np.round(np.random.exponential(scale=10.0, size=num_samples), 1),
        "device_screen_resolution": random.choices(["1125x2436", "1920x1080", "1440x2560"], k=num_samples),
        "device_language": ["en-US"] * num_samples,
        "device_timezone": random.choices(["America/New_York", "America/Los_Angeles", "America/Chicago"], k=num_samples),
        "device_fonts_hash": [f"FONT_{random.randint(1000,9999)}" for _ in range(num_samples)],
        
        # Transaction Timing
        "transaction_hour": [ts.hour for ts in timestamps],
        "transaction_day_of_week": [ts.weekday() for ts in timestamps],
        "transaction_day_of_month": [ts.day for ts in timestamps],
        "transaction_day_of_year": [ts.timetuple().tm_yday for ts in timestamps],
        "days_since_last_transaction": np.random.randint(0, 30, size=num_samples),
        "seconds_since_last_login": np.random.randint(0, 3600, size=num_samples),
        "transaction_duration_sec": np.round(np.random.exponential(scale=30.0, size=num_samples), 1),
        
        # Behavioral Features
        "transactions_last_1h": np.random.poisson(lam=0.5, size=num_samples),
        "transactions_last_24h": np.random.poisson(lam=3.0, size=num_samples),
        "transactions_last_7d": np.random.poisson(lam=15.0, size=num_samples),
        "transactions_last_30d": np.random.poisson(lam=60.0, size=num_samples),
        "avg_transaction_gap_1w_sec": np.round(np.random.normal(loc=86400, scale=21600, size=num_samples), 0),
        "user_typical_transaction_hour": np.random.randint(0, 24, size=num_samples),
        "user_typical_transaction_amount": np.round(np.random.lognormal(mean=4.0, sigma=0.5, size=num_samples), 2),
        "user_typical_transaction_location": random.choices(["home", "work", "travel"], k=num_samples),
        
        # Payment Method
        "payment_type": random.choices(["credit_card", "debit_card", "paypal"], weights=[0.7, 0.2, 0.1], k=num_samples),
        "card_brand": random.choices(["VISA", "MasterCard", "Amex"], weights=[0.6, 0.3, 0.1], k=num_samples),
        "card_type": random.choices(["standard", "gold", "platinum"], weights=[0.6, 0.3, 0.1], k=num_samples),
        "card_expiry_months": np.random.randint(1, 36, size=num_samples),
        "card_tokenized": np.random.binomial(n=1, p=0.8, size=num_samples),
        "card_3ds_authenticated": np.random.binomial(n=1, p=0.7, size=num_samples),
        "card_issuer": random.choices(["Chase", "Bank of America", "Citi", "Wells Fargo"], k=num_samples),
        "card_issuer_country": ["US"] * num_samples,
        "payment_processor": random.choices(["Stripe", "PayPal", "Square"], k=num_samples),
        "payment_authentication_method": random.choices(["3ds", "none", "biometric"], weights=[0.7, 0.2, 0.1], k=num_samples),
        
        # Network Features
        "ip_risk_score": np.round(np.random.beta(a=1, b=9, size=num_samples), 4),
        "ip_proxy_used": np.random.binomial(n=1, p=0.05, size=num_samples),
        "ip_asn": [f"AS{random.randint(1000,9999)}" for _ in range(num_samples)],
        "ip_distance_km": np.round(np.random.exponential(scale=50.0, size=num_samples), 1),
        "ip_city_match": np.random.binomial(n=1, p=0.9, size=num_samples),
        "ip_country_match": np.random.binomial(n=1, p=0.95, size=num_samples),
        "ip_isp": random.choices(["Comcast", "AT&T", "Verizon", "T-Mobile"], k=num_samples),
        "ip_connection_type": random.choices(["broadband", "mobile", "dialup"], weights=[0.8, 0.19, 0.01], k=num_samples),
        "ip_anonymous_proxy": np.random.binomial(n=1, p=0.01, size=num_samples),
        "ip_reputation_score": np.round(np.random.beta(a=9, b=1, size=num_samples), 2),
        
        # Transaction Context
        "basket_size": np.random.randint(1, 10, size=num_samples),
        "contains_digital_goods": np.random.binomial(n=1, p=0.3, size=num_samples),
        "contains_high_risk_items": np.random.binomial(n=1, p=0.1, size=num_samples),
        "same_as_shipping_address": np.random.binomial(n=1, p=0.8, size=num_samples),
        "shipping_speed": random.choices(["standard", "express", "overnight"], weights=[0.7, 0.2, 0.1], k=num_samples),
        "order_amount_usd": np.round(np.random.lognormal(mean=4.5, sigma=0.8, size=num_samples), 2),
        "order_currency": ["USD"] * num_samples,
        "order_discount_amount": np.round(np.random.exponential(scale=5.0, size=num_samples), 2),
        "order_has_coupon": np.random.binomial(n=1, p=0.2, size=num_samples),
        "order_item_count": np.random.randint(1, 10, size=num_samples),
        
        # User Behavior
        "session_duration_sec": np.round(np.random.exponential(scale=120.0, size=num_samples), 1),
        "mouse_movements": np.random.poisson(lam=25.0, size=num_samples),
        "keystroke_speed": np.round(np.random.normal(loc=4.0, scale=1.0, size=num_samples), 1),
        "login_attempts": np.random.poisson(lam=1.2, size=num_samples),
        "page_activity_score": np.round(np.random.beta(a=2, b=2, size=num_samples), 2),
        "form_fill_time": np.round(np.random.exponential(scale=15.0, size=num_samples), 1),
        "click_pattern_score": np.round(np.random.beta(a=3, b=1, size=num_samples), 2),
        "mouse_velocity": np.round(np.random.exponential(scale=1.0, size=num_samples), 1),
        "scroll_behavior": random.choices(["steady", "erratic", "none"], weights=[0.7, 0.2, 0.1], k=num_samples),
        "page_transitions": np.random.poisson(lam=3.0, size=num_samples),
        
        # Merchant-side Data
        "merchant_avs_response": random.choices(["Y", "N", "A"], weights=[0.8, 0.1, 0.1], k=num_samples),
        "merchant_cvv_response": random.choices(["M", "N", "P"], weights=[0.9, 0.05, 0.05], k=num_samples),
        "merchant_risk_decision": random.choices(["accept", "review", "decline"], weights=[0.85, 0.1, 0.05], k=num_samples),
        "merchant_fraud_filters_triggered": np.random.poisson(lam=0.2, size=num_samples),
        "merchant_decision_time_ms": np.random.randint(50, 500, size=num_samples),
        "merchant_score": np.round(np.random.beta(a=8, b=2, size=num_samples), 2),
        "merchant_velocity_check": np.random.binomial(n=1, p=0.9, size=num_samples),
        "merchant_blacklist_check": np.random.binomial(n=1, p=0.05, size=num_samples),
        "merchant_whitelist_check": np.random.binomial(n=1, p=0.1, size=num_samples),
        "merchant_custom_rules_triggered": np.random.poisson(lam=0.1, size=num_samples),
        
        # Historical Patterns
        "user_hist_chargeback_rate": np.round(np.random.beta(a=1, b=99, size=num_samples), 4),
        "user_hist_merchant_transactions": np.random.poisson(lam=5.0, size=num_samples),
        "user_hist_category_transactions": np.random.poisson(lam=20.0, size=num_samples),
        "user_hist_amount_deviation": np.round(np.random.exponential(scale=0.2, size=num_samples), 2),
        "user_hist_time_deviation": np.round(np.random.exponential(scale=0.3, size=num_samples), 2),
        "user_hist_location_deviation": np.round(np.random.exponential(scale=0.1, size=num_samples), 2),
        "user_hist_device_consistency": np.round(np.random.beta(a=9, b=1, size=num_samples), 2),
        "user_hist_payment_consistency": np.round(np.random.beta(a=9, b=1, size=num_samples), 2),
        "user_hist_behavior_score": np.round(np.random.beta(a=8, b=2, size=num_samples), 2),
        "user_hist_risk_score": np.round(np.random.beta(a=1, b=9, size=num_samples), 2),
        
        # Derived Features
        "amount_to_avg_balance_ratio": np.round(np.random.exponential(scale=0.01, size=num_samples), 3),
        "location_velocity_kmh": np.round(np.random.exponential(scale=10.0, size=num_samples), 1),
        "transaction_size_percentile": np.round(np.random.uniform(low=0.0, high=1.0, size=num_samples), 2),
        "time_of_day_risk_score": np.round(np.random.beta(a=1, b=9, size=num_samples), 2),
        "device_trust_score": np.round(np.random.beta(a=9, b=1, size=num_samples), 2),
        "behavioral_anomaly_score": np.round(np.random.beta(a=1, b=9, size=num_samples), 2),
        "network_risk_score": np.round(np.random.beta(a=1, b=9, size=num_samples), 2),
        "payment_anomaly_score": np.round(np.random.beta(a=1, b=9, size=num_samples), 2),
        "temporal_anomaly_score": np.round(np.random.beta(a=1, b=9, size=num_samples), 2),
        "composite_risk_score": np.round(np.random.beta(a=1, b=9, size=num_samples), 2),
        
        # Supplemental Data
        "user_email_domain": random.choices(["gmail.com", "yahoo.com", "outlook.com"], k=num_samples),
        "user_phone_carrier": random.choices(["Verizon", "AT&T", "T-Mobile"], k=num_samples),
        "user_social_media_linked": np.random.binomial(n=1, p=0.7, size=num_samples),
        "user_2fa_enabled": np.random.binomial(n=1, p=0.5, size=num_samples),
        "user_has_verified_identity": np.random.binomial(n=1, p=0.8, size=num_samples),
        "user_email_age_days": np.random.randint(30, 365*10, size=num_samples),
        "user_phone_age_days": np.random.randint(30, 365*5, size=num_samples),
        "user_account_verified": np.random.binomial(n=1, p=0.9, size=num_samples),
        "user_kyc_level": np.random.randint(1, 4, size=num_samples),
        "user_reputation_score": np.round(np.random.beta(a=9, b=1, size=num_samples), 2),
        
        # Merchant Technical
        "merchant_checkout_version": random.choices(["v2.1", "v3.0", "v3.2"], k=num_samples),
        "merchant_platform": random.choices(["shopify", "magento", "custom"], k=num_samples),
        "merchant_tls_version": random.choices(["1.2", "1.3"], weights=[0.3, 0.7], k=num_samples),
        "merchant_pci_compliant": np.random.binomial(n=1, p=0.95, size=num_samples),
        "merchant_fraud_tools": random.choices(["kount", "sift", "none"], weights=[0.4, 0.3, 0.3], k=num_samples),
        "merchant_authentication_methods": random.choices(["3ds", "biometric", "none"], weights=[0.7, 0.2, 0.1], k=num_samples),
        "merchant_checkout_flow": random.choices(["one_page", "multi_step"], weights=[0.6, 0.4], k=num_samples),
        "merchant_anti_fraud_system": random.choices(["internal", "external", "none"], weights=[0.5, 0.3, 0.2], k=num_samples),
        "merchant_decision_engine": random.choices(["rules_v2", "rules_v3", "ml_model"], weights=[0.3, 0.5, 0.2], k=num_samples),
        "merchant_data_quality_score": np.round(np.random.beta(a=9, b=1, size=num_samples), 2),
        
        # Temporal Patterns
        "hourly_transaction_velocity": np.round(np.random.exponential(scale=0.3, size=num_samples), 2),
        "weekly_pattern_match": np.round(np.random.beta(a=8, b=2, size=num_samples), 2),
        "holiday_indicator": np.random.binomial(n=1, p=0.05, size=num_samples),
        "seasonal_risk_factor": np.round(np.random.beta(a=1, b=9, size=num_samples), 2),
        "time_since_last_chargeback": np.random.randint(0, 365, size=num_samples),
        "time_since_last_fraud": np.random.randint(0, 365, size=num_samples),
        "transaction_time_deviation": np.round(np.random.exponential(scale=0.2, size=num_samples), 2),
        "user_activity_time_score": np.round(np.random.beta(a=8, b=2, size=num_samples), 2),
        "merchant_peak_hour": np.random.binomial(n=1, p=0.3, size=num_samples),
        "time_based_risk_score": np.round(np.random.beta(a=1, b=9, size=num_samples), 2),
        
        # Cross-feature Interactions
        "high_value_new_device_score": np.round(np.random.beta(a=1, b=9, size=num_samples), 2),
        "category_velocity_alert": np.random.binomial(n=1, p=0.05, size=num_samples),
        "geo_ip_mismatch_score": np.round(np.random.beta(a=1, b=9, size=num_samples), 2),
        "browser_os_consistency": np.random.binomial(n=1, p=0.9, size=num_samples),
        "device_payment_mismatch": np.random.binomial(n=1, p=0.1, size=num_samples),
        "shipping_billing_discrepancy": np.random.binomial(n=1, p=0.1, size=num_samples),
        "behavioral_payment_anomaly": np.round(np.random.beta(a=1, b=9, size=num_samples), 2),
        "temporal_location_risk": np.round(np.random.beta(a=1, b=9, size=num_samples), 2),
        "user_merchant_trust_score": np.round(np.random.beta(a=8, b=2, size=num_samples), 2),
        "composite_interaction_score": np.round(np.random.beta(a=1, b=9, size=num_samples), 2),
        
        # Engineered Features (50 total)
        **{f"engineered_feature_{i}": np.round(np.random.uniform(low=0.0, high=1.0, size=num_samples), 2) 
           for i in range(1, 51)},
        
        # Target Variable (Fraud Indicator)
        "is_fraud": np.random.binomial(n=1, p=fraud_ratio, size=num_samples)
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data(num_samples=1000, fraud_ratio=0.01)
    
    # Ensure directories exist
    settings.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(settings.RAW_DATA_PATH, index=False)
    print(f"Successfully generated sample data with {len(df)} records at:")
    print(settings.RAW_DATA_PATH)
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")