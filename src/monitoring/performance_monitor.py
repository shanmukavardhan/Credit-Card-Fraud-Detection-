import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import precision_score, recall_score, f1_score
from config.settings import settings
import logging

logging.config.fileConfig(settings.LOGGING_CONFIG, disable_existing_loggers=False)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self, window_size: int = None):
        self.window_size = window_size or settings.MONITORING_WINDOW_SIZE
        self.predictions = []
        self.actuals = []
        
    def add_prediction(self, prediction: Dict[str, Any], actual: int = None) -> None:
        """Add a prediction to the monitoring buffer"""
        self.predictions.append(prediction)
        if actual is not None:
            self.actuals.append(actual)
        
        # Maintain window size
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            if len(self.actuals) > 0:
                self.actuals.pop(0)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics over the window"""
        if len(self.actuals) == 0:
            return {}
            
        pred_classes = [p['is_fraud'] for p in self.predictions[-len(self.actuals):]]
        pred_probas = [p['probability'] for p in self.predictions[-len(self.actuals):]]
        
        return {
            'precision': precision_score(self.actuals, pred_classes),
            'recall': recall_score(self.actuals, pred_classes),
            'f1': f1_score(self.actuals, pred_classes),
            'num_samples': len(self.actuals)
        }
    
    def check_for_drift(self, reference_metrics: Dict[str, float], threshold: float = None) -> Dict[str, bool]:
        """Check for performance drift compared to reference metrics"""
        if threshold is None:
            threshold = settings.DRIFT_THRESHOLD
            
        current_metrics = self.calculate_metrics()
        drift_detected = {}
        
        for metric in ['precision', 'recall', 'f1']:
            if metric in reference_metrics and metric in current_metrics:
                relative_change = abs((current_metrics[metric] - reference_metrics[metric]) / reference_metrics[metric])
                drift_detected[metric] = relative_change > threshold
        
        return drift_detected