# anomaly_detection/__init__.py
"""
Anomaly Detection Package
Integrated detection using Rule-Based, Statistical, and ML methods
"""

from .data_preparation import AttendanceDataPreprocessor, prepare_attendance_data
from .rule_based_detector import RuleBasedAnomalyDetector, detect_anomalies
from .statistical_detector import AdaptiveStatisticalDetector, detect_statistical_anomalies
from .ml_detector import MLAnomalyDetector, create_ml_detector

def __init__(self,
             model_dir: str = 'models/ml_detector',
             contamination: float = 0.1,
             adaptation_rate: float = 0.2,
             retrain_interval_days: int = 7,
             min_samples_retrain: int = 50,
             max_features: int = 50,
             use_threading_timeout: bool = False):  # Add this parameter
    """
    Initialize ML anomaly detector.
    """
    self.model_dir = model_dir
    self.contamination = contamination
    self.adaptation_rate = adaptation_rate
    self.retrain_interval_days = retrain_interval_days
    self.min_samples_retrain = min_samples_retrain
    self.max_features = max_features
    self.use_threading_timeout = use_threading_timeout  # Store this
    
    # ... rest of your __init__ code ...
    
__version__ = "1.0.0"
__all__ = [
    'AttendanceDataPreprocessor',
    'prepare_attendance_data',
    'RuleBasedAnomalyDetector', 
    'detect_anomalies',
    'AdaptiveStatisticalDetector',
    'detect_statistical_anomalies',
    'MLAnomalyDetector',
    'create_ml_detector'
]