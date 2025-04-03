from datetime import datetime

class TransactionFeatureEngineer:
    def __init__(self):
        # Load any pre-trained transformers
        self.scaler = joblib.load(settings.SCALER_PATH)
        self.pca = joblib.load(settings.PCA_PATH)
        
        # Merchant risk profiles
        self.merchant_risk = {
            'gas_station': 1.2,
            'online_retail': 1.5,
            'electronics': 1.8,
            'travel': 2.0,
            'groceries': 0.8
        }

    def engineer_features(self, transaction_data: dict) -> dict:
        """Convert raw transaction data to V1-V28 features"""
        # 1. Basic features
        features = {
            'time': self._get_transaction_time(),
            'amount': transaction_data['amount'],
            'amount_log': np.log1p(transaction_data['amount']),
            'is_foreign': int(transaction_data['country'] != 'US'),
            'merchant_risk': self.merchant_risk.get(
                transaction_data['merchant_type'], 1.0),
            'is_night': int(0 <= datetime.now().hour < 6),
            'customer_history': transaction_data.get('customer_history', 0)
        }

        # 2. Create derived features
        features.update({
            'amount_per_hist': features['amount'] / (features['customer_history'] + 1),
            'time_since_last': transaction_data.get('time_since_last', 0)
        })

        # 3. Normalize and transform to V features
        base_features = pd.DataFrame([features])[['amount_log', 'is_foreign', 
                                               'merchant_risk', 'is_night']]
        
        # Scale features
        scaled = self.scaler.transform(base_features)
        
        # Apply PCA to get V features
        v_features = self.pca.transform(scaled)
        
        # Create output dictionary
        output = {'Time': features['time'], 'Amount': features['amount']}
        for i in range(1, 29):
            output[f'V{i}'] = v_features[0][i-1] if i-1 < len(v_features[0]) else 0.0
        
        return output

    def _get_transaction_time(self):
        """Convert current time to seconds since first transaction of day"""
        now = datetime.now()
        return (now.hour * 3600 + now.minute * 60 + now.second)