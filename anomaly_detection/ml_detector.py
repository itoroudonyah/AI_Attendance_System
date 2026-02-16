"""
Machine Learning Anomaly Detection Module
Primary method: Isolation Forest with adaptive learning
Secondary methods: Autoencoders, Time-series models
Production-ready with model persistence and monitoring
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import warnings
import signal
import threading
from contextlib import contextmanager

warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA

# Utility
import joblib
import os
import json

class MLAnomalyDetector:
    """
    Machine Learning based anomaly detection system.
    Primary method: Adaptive Isolation Forest
    Features: Model persistence, adaptive learning, explainability
    """
    
    def __init__(self,
                 model_dir: str = 'models/ml_detector',
                 contamination: float = 0.1,
                 adaptation_rate: float = 0.2,
                 retrain_interval_days: int = 7,
                 min_samples_retrain: int = 50,
                 max_features: int = 50):
        """
        Initialize ML anomaly detector.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save/load models
        contamination : float
            Expected proportion of outliers (0.01 to 0.3)
        adaptation_rate : float
            How quickly models adapt to new patterns (0.0 to 1.0)
        retrain_interval_days : int
            Days between full retraining
        min_samples_retrain : int
            Minimum new samples before incremental update
        max_features : int
            Maximum number of features to use (prevent explosion)
        """
        self.model_dir = model_dir
        self.contamination = contamination
        self.adaptation_rate = adaptation_rate
        self.retrain_interval_days = retrain_interval_days
        self.min_samples_retrain = min_samples_retrain
        self.max_features = max_features
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)
        
        # Primary model: Isolation Forest
        self.isolation_forest = None
        
        # Supporting models
        self.autoencoder = None
        self.lstm_model = None
        
        # Feature processors
        self.scaler_robust = RobustScaler()
        self.scaler_standard = StandardScaler()
        self.pca = None
        
        # Feature engineering
        self.feature_columns = None
        self.feature_importance = {}
        self.training_mode = None
        
        # Adaptive learning
        self.new_data_buffer = []
        self.last_retrain_date = None
        self.performance_history = []
        self.model_metadata = {}

        # Load existing models
        self._load_models()

    @contextmanager
    def _timeout(self, seconds: int):
        """Timeout context manager."""
        if threading.current_thread() is not threading.main_thread():
            yield
            return

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")
        
        # Set the timeout signal
        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            # Disable the alarm
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

    def _sanitize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Optimized feature sanitization for large datasets."""
        sanitized = features_df.copy()
        
        # Process in chunks for large DataFrames
        if len(sanitized) > 1000:
            # Vectorized operations for numeric columns
            numeric_cols = sanitized.select_dtypes(include=["number"]).columns
            
            if len(numeric_cols) > 0:
                # Replace inf with nan
                sanitized[numeric_cols] = sanitized[numeric_cols].replace([np.inf, -np.inf], np.nan)
                
                # Fill NaN with median in one operation
                for col in numeric_cols:
                    median_val = sanitized[col].median()
                    sanitized[col] = sanitized[col].fillna(median_val)
            
            # Non-numeric columns
            non_numeric_cols = sanitized.columns.difference(numeric_cols)
            for col in non_numeric_cols:
                sanitized[col] = sanitized[col].fillna(0)
        else:
            # Original logic for small datasets
            sanitized.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            numeric_cols = sanitized.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                sanitized[col] = sanitized[col].fillna(sanitized[col].median())
            
            non_numeric_cols = sanitized.columns.difference(numeric_cols)
            for col in non_numeric_cols:
                sanitized[col] = sanitized[col].fillna(0)
        
        return sanitized
    
    def _load_models(self):
        """Load existing trained models from disk."""
        models_to_load = {
            'isolation_forest': 'isolation_forest.pkl',
            'scaler_robust': 'scaler_robust.pkl',
            'scaler_standard': 'scaler_standard.pkl',
            'pca': 'pca.pkl',
            'feature_columns': 'feature_columns.pkl',
            'metadata': 'metadata.json'
        }
        
        for model_name, filename in models_to_load.items():
            model_path = os.path.join(self.model_dir, filename)
            
            if os.path.exists(model_path):
                try:
                    if filename.endswith('.json'):
                        with open(model_path, 'r') as f:
                            setattr(self, f'{model_name}_metadata', json.load(f))
                    else:
                        model = joblib.load(model_path)
                        setattr(self, model_name, model)
                        print(f"✓ Loaded {model_name} from {model_path}")
                except Exception as e:
                    print(f"✗ Error loading {model_name}: {e}")
    
    def _save_models(self):
        """Save all models and metadata to disk."""
        models_to_save = {
            'isolation_forest': self.isolation_forest,
            'scaler_robust': self.scaler_robust,
            'scaler_standard': self.scaler_standard,
            'pca': self.pca,
            'feature_columns': self.feature_columns
        }
        
        # --- PCA compatibility guard ---
        if getattr(self, "pca", None) is not None:
            expected = getattr(self.pca, "n_features_in_", None)

            # Some sklearn versions may not have n_features_in_
            if expected is None and hasattr(self.pca, "components_"):
                expected = self.pca.components_.shape[1]

            current = None
            if getattr(self, "feature_columns", None) is not None:
                current = len(self.feature_columns)

            if expected is not None and current is not None and expected != current:
                print(f"⚠️ PCA disabled: PCA expects {expected} features, but current feature_columns has {current}.")
                self.pca = None
                self.feature_columns = None
    

        for model_name, model in models_to_save.items():
            if model is not None:
                model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
                joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'last_retrain_date': self.last_retrain_date.isoformat() if self.last_retrain_date else None,
            'contamination': self.contamination,
            'adaptation_rate': self.adaptation_rate,
            'performance_history': self.performance_history[-10:],  # Keep last 10
            'feature_importance': self.feature_importance,
            'max_features': self.max_features
        }
        
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _create_essential_features(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create essential features for large datasets.
        """
        df = daily_df.copy()
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values(['employee_id', 'date'])
        
        # Only create essential features (no rolling, no lag)
        essential_features = []
        
        # 1. Basic time features
        if 'arrival_total_minutes' in df.columns:
            df['arrival_sin'] = np.sin(2 * np.pi * df['arrival_total_minutes'] / (24 * 60))
            df['arrival_cos'] = np.cos(2 * np.pi * df['arrival_total_minutes'] / (24 * 60))
            essential_features.extend(['arrival_sin', 'arrival_cos'])
        
        if 'departure_total_minutes' in df.columns:
            df['departure_sin'] = np.sin(2 * np.pi * df['departure_total_minutes'] / (24 * 60))
            df['departure_cos'] = np.cos(2 * np.pi * df['departure_total_minutes'] / (24 * 60))
            essential_features.extend(['departure_sin', 'departure_cos'])
        
        # 2. Simple statistical features (no rolling calculations)
        if 'work_duration_hours' in df.columns:
            df['work_duration_z'] = df.groupby('employee_id')['work_duration_hours'].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6)
            )
            essential_features.append('work_duration_z')
        
        if 'lateness_minutes' in df.columns:
            df['lateness_abs'] = df['lateness_minutes'].abs()
            essential_features.append('lateness_abs')
        
        # 3. Interaction features
        if all(col in df.columns for col in ['work_duration_hours', 'total_checks']):
            df['checks_per_hour'] = df['total_checks'] / (df['work_duration_hours'] + 1e-6)
            essential_features.append('checks_per_hour')
        
        # 4. Day of week features
        if 'arrival_total_minutes' in df.columns and 'day_of_week' in df.columns:
            df['dow_arrival'] = df.groupby('day_of_week')['arrival_total_minutes'].transform('mean')
            essential_features.append('dow_arrival')
        
        # 5. Department features
        if 'department' in df.columns and 'work_duration_hours' in df.columns:
            df['dept_hours_mean'] = df.groupby('department')['work_duration_hours'].transform('mean')
            df['dept_hours_dev'] = df['work_duration_hours'] - df['dept_hours_mean']
            essential_features.extend(['dept_hours_mean', 'dept_hours_dev'])
        
        # 6. Basic location features
        if 'unique_locations' in df.columns:
            df['has_multiple_locs'] = (df['unique_locations'] > 1).astype(int)
            essential_features.append('has_multiple_locs')
        
        if 'location_changes' in df.columns:
            df['location_changes_norm'] = df['location_changes'] / (df['work_duration_hours'] + 1e-6)
            essential_features.append('location_changes_norm')
        
        # 7. Sequence features
        if 'has_incomplete_pairs' in df.columns:
            df['incomplete_pairs_flag'] = df['has_incomplete_pairs'].astype(int)
            essential_features.append('incomplete_pairs_flag')
        
        if 'sequence_issues_count' in df.columns:
            df['seq_issues_norm'] = df['sequence_issues_count'] / (df['total_checks'] + 1e-6)
            essential_features.append('seq_issues_norm')
        
        # Return only essential features that exist
        existing_features = [f for f in essential_features if f in df.columns]
        
        # Limit features if too many
        if len(existing_features) > self.max_features:
            print(f"Limiting essential features from {len(existing_features)} to {self.max_features}")
            
            # Keep most important features if available
            if self.feature_importance:
                # Sort features by importance
                important_features = sorted(
                    self.feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:self.max_features]
                important_feature_names = [f[0] for f in important_features]
                
                # Keep intersection with existing features
                selected_features = [
                    f for f in important_feature_names 
                    if f in existing_features
                ][:self.max_features]
            else:
                # Select first max_features features
                selected_features = existing_features[:self.max_features]
        else:
            selected_features = existing_features
        
        features_df = df[selected_features].copy()
        
        return self._sanitize_features(features_df)
    
    def create_advanced_features(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced ML features for anomaly detection.
        
        Args:
            daily_df: Daily attendance summaries
            
        Returns:
            DataFrame with engineered features
        """
        # Early return for large datasets - use optimized features
        if len(daily_df) > 1000:
            print(f"Large dataset detected ({len(daily_df)} rows). Using optimized feature engineering...")
            return self._create_essential_features(daily_df)
        
        df = daily_df.copy()
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date for time-based features
        df = df.sort_values(['employee_id', 'date'])
        
        # 1. Time-based features
        time_features = []
        
        # Cyclical time features
        if 'arrival_total_minutes' in df.columns:
            df['arrival_sin'] = np.sin(2 * np.pi * df['arrival_total_minutes'] / (24 * 60))
            df['arrival_cos'] = np.cos(2 * np.pi * df['arrival_total_minutes'] / (24 * 60))
            time_features.extend(['arrival_sin', 'arrival_cos'])
        
        if 'departure_total_minutes' in df.columns:
            df['departure_sin'] = np.sin(2 * np.pi * df['departure_total_minutes'] / (24 * 60))
            df['departure_cos'] = np.cos(2 * np.pi * df['departure_total_minutes'] / (24 * 60))
            time_features.extend(['departure_sin', 'departure_cos'])
        
        # 2. Statistical features per employee
        statistical_features = []
        
        for emp_id in df['employee_id'].unique():
            emp_mask = df['employee_id'] == emp_id
            emp_data = df[emp_mask].copy()
            
            if len(emp_data) < 3:
                continue
            
            # Rolling statistics
            for col in ['work_duration_hours', 'lateness_minutes', 'arrival_total_minutes']:
                if col in emp_data.columns:
                    # Rolling mean and std
                    df.loc[emp_mask, f'{col}_rolling_mean_7'] = emp_data[col].rolling(7, min_periods=3).mean()
                    df.loc[emp_mask, f'{col}_rolling_std_7'] = emp_data[col].rolling(7, min_periods=3).std()
                    
                    # Deviation from rolling mean
                    df.loc[emp_mask, f'{col}_deviation'] = (
                        emp_data[col] - emp_data[col].rolling(7, min_periods=3).mean()
                    ) / (emp_data[col].rolling(7, min_periods=3).std() + 1e-6)
                    
                    statistical_features.extend([
                        f'{col}_rolling_mean_7',
                        f'{col}_rolling_std_7',
                        f'{col}_deviation'
                    ])
            
            # Rate of change
            for col in ['work_duration_hours', 'arrival_total_minutes']:
                if col in emp_data.columns:
                    df.loc[emp_mask, f'{col}_roc'] = emp_data[col].pct_change()
                    statistical_features.append(f'{col}_roc')
        
        # 3. Interaction features
        interaction_features = []
        
        if all(col in df.columns for col in ['work_duration_hours', 'lateness_minutes']):
            df['efficiency_score'] = df['work_duration_hours'] / (df['lateness_minutes'].abs() + 1)
            interaction_features.append('efficiency_score')
        
        if all(col in df.columns for col in ['work_duration_hours', 'total_checks']):
            df['checks_per_hour'] = df['total_checks'] / (df['work_duration_hours'] + 1e-6)
            interaction_features.append('checks_per_hour')
        
        if all(col in df.columns for col in ['unique_locations', 'work_duration_hours']):
            df['location_intensity'] = df['unique_locations'] / (df['work_duration_hours'] + 1e-6)
            interaction_features.append('location_intensity')
        
        # 4. Pattern consistency features
        pattern_features = []
        
        # Day of week patterns
        if 'arrival_total_minutes' in df.columns and 'day_of_week' in df.columns:
            df['dow_arrival_mean'] = df.groupby(['employee_id', 'day_of_week'])['arrival_total_minutes'].transform('mean')
            df['dow_arrival_std'] = df.groupby(['employee_id', 'day_of_week'])['arrival_total_minutes'].transform('std')
            pattern_features.extend(['dow_arrival_mean', 'dow_arrival_std'])
        
        # Consistency score (how consistent is the employee)
        if 'dow_arrival_std' in df.columns:
            df['consistency_score'] = 1 / (df['dow_arrival_std'] + 1)
            pattern_features.append('consistency_score')
        
        # 5. Lag features (previous day behavior)
        lag_features = []
        
        for emp_id in df['employee_id'].unique():
            emp_mask = df['employee_id'] == emp_id
            emp_data = df[emp_mask].copy()
            
            if len(emp_data) < 2:
                continue
            
            for col in ['work_duration_hours', 'lateness_minutes', 'arrival_total_minutes']:
                if col in emp_data.columns:
                    df.loc[emp_mask, f'{col}_lag1'] = emp_data[col].shift(1)
                    df.loc[emp_mask, f'{col}_lag2'] = emp_data[col].shift(2)
                    lag_features.extend([f'{col}_lag1', f'{col}_lag2'])
        
        # 6. Aggregated features
        aggregated_features = []
        
        # Employee overall statistics
        for col in ['work_duration_hours', 'lateness_minutes', 'arrival_total_minutes']:
            if col in df.columns:
                df[f'{col}_emp_mean'] = df.groupby('employee_id')[col].transform('mean')
                df[f'{col}_emp_std'] = df.groupby('employee_id')[col].transform('std')
                aggregated_features.extend([f'{col}_emp_mean', f'{col}_emp_std'])
        
        # Department statistics
        if 'department' in df.columns:
            for col in ['work_duration_hours', 'lateness_minutes']:
                if col in df.columns:
                    df[f'{col}_dept_mean'] = df.groupby('department')[col].transform('mean')
                    df[f'{col}_dept_std'] = df.groupby('department')[col].transform('std')
                    aggregated_features.extend([f'{col}_dept_mean', f'{col}_dept_std'])
        
        # Combine all features
        all_features = time_features + statistical_features + interaction_features + \
                      pattern_features + lag_features + aggregated_features
        
        # Only keep features that exist
        existing_features = [f for f in all_features if f in df.columns]
        
        # Limit features if too many
        if len(existing_features) > self.max_features:
            print(f"Limiting advanced features from {len(existing_features)} to {self.max_features}")
            
            # Keep most important features if available
            if self.feature_importance:
                # Sort features by importance
                important_features = sorted(
                    self.feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:self.max_features]
                important_feature_names = [f[0] for f in important_features]
                
                # Keep intersection with existing features
                selected_features = [
                    f for f in important_feature_names 
                    if f in existing_features
                ][:self.max_features]
            else:
                # Select first max_features features
                selected_features = existing_features[:self.max_features]
        else:
            selected_features = existing_features
        
        features_df = df[selected_features].copy()
        
        return self._sanitize_features(features_df)
    
    def train_isolation_forest(self, daily_df: pd.DataFrame, 
                              n_estimators: int = 100,
                              max_samples: Union[str, float] = 'auto',
                              random_state: int = 42,
                              use_essential_features: bool = False) -> Dict[str, Any]:
        """
        Train Isolation Forest model.
        
        Args:
            daily_df: Daily attendance summaries
            n_estimators: Number of trees in the forest
            max_samples: Number of samples to draw for each tree
            random_state: Random seed
            
        Returns:
            Training results dictionary
        """
        print("Training Isolation Forest model...")
        
        # Create features
        if use_essential_features:
            features_df = self._create_essential_features(daily_df)
            self.training_mode = "essential"
        else:
            features_df = self.create_advanced_features(daily_df)
            self.training_mode = "advanced"
        self.feature_columns = features_df.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler_robust.fit_transform(features_df)
        
        # Train PCA for dimensionality reduction (optional)
        if features_df.shape[1] > 20:
            self.pca = PCA(n_components=0.95, random_state=random_state)
            X_transformed = self.pca.fit_transform(X_scaled)
            print(f"PCA reduced features from {features_df.shape[1]} to {self.pca.n_components_}")
        else:
            X_transformed = X_scaled
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=self.contamination,
            random_state=random_state,
            n_jobs=-1,
            verbose=1 if len(daily_df) > 1000 else 0
        )
        
        self.isolation_forest.fit(X_transformed)
        
        # Calculate feature importance
        self._calculate_feature_importance(features_df, X_transformed)
        
        # Evaluate on training data
        train_predictions = self.isolation_forest.predict(X_transformed)
        train_scores = self.isolation_forest.score_samples(X_transformed)
        
        # Convert scores to anomaly probabilities
        train_anomaly_probs = self._scores_to_probabilities(train_scores)
        
        # Update metadata
        self.last_retrain_date = datetime.now()
        
        # Record performance
        performance = {
            'timestamp': datetime.now().isoformat(),
            'model': 'isolation_forest',
            'n_samples': len(daily_df),
            'n_features': X_transformed.shape[1],
            'contamination': self.contamination,
            'anomaly_rate': (train_predictions == -1).mean(),
            'avg_anomaly_score': train_anomaly_probs.mean(),
            'top_features': list(self.feature_importance.items())[:5]
        }
        
        self.performance_history.append(performance)
        
        # Save models
        self._save_models()
        
        print("✓ Isolation Forest training complete!")
        print(f"  - Samples: {len(daily_df)}")
        print(f"  - Features: {X_transformed.shape[1]}")
        print(f"  - Detected anomalies: {(train_predictions == -1).sum()} ({performance['anomaly_rate']:.1%})")
        
        return performance
    
    def _calculate_feature_importance(self, features_df: pd.DataFrame, 
                                     X_transformed: np.ndarray):
        """Calculate feature importance for Isolation Forest."""
        if self.isolation_forest is None:
            return
        
        # Get feature importances from Isolation Forest
        if hasattr(self.isolation_forest, 'feature_importances_'):
            importances = self.isolation_forest.feature_importances_
        else:
            # Estimate importance using permutation
            importances = np.zeros(X_transformed.shape[1])
            baseline_score = self.isolation_forest.score_samples(X_transformed).mean()
            
            for i in range(X_transformed.shape[1]):
                X_permuted = X_transformed.copy()
                np.random.shuffle(X_permuted[:, i])
                permuted_score = self.isolation_forest.score_samples(X_permuted).mean()
                importances[i] = baseline_score - permuted_score
        
        # Map back to original features if PCA was used
        if (
            self.pca is not None
            and self.pca.components_ is not None
            and importances.shape[0] == self.pca.components_.shape[0]
        ):
            # Transform importance back to original feature space
            original_importances = np.abs(self.pca.components_).T.dot(importances)
            
            # Normalize
            if original_importances.sum() > 0:
                original_importances = original_importances / original_importances.sum()
            
            # Create dictionary
            self.feature_importance = {
                feature: importance 
                for feature, importance in zip(self.feature_columns, original_importances)
            }
        else:
            # Direct mapping
            self.feature_importance = {
                feature: importance 
                for feature, importance in zip(self.feature_columns, importances)
            }
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
    
    def _scores_to_probabilities(self, scores: np.ndarray) -> np.ndarray:
        """Convert Isolation Forest scores to anomaly probabilities (0-1)."""
        # Isolation Forest returns negative scores for anomalies
        # Lower score = more anomalous
        
        # Normalize scores to [0, 1] where 1 is most anomalous
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        
        # Invert so higher = more anomalous
        anomaly_probs = 1 - scores_normalized
        
        return anomaly_probs
    
    def _detect_anomalies_impl(self, daily_df: pd.DataFrame, 
                              probability_threshold: float = 0.7,
                              use_advanced_features: bool = True,
                              include_contributions: bool = False,
                              enable_adaptive: bool = True,
                              debug_steps: bool = False) -> pd.DataFrame:
        """
        Implementation of anomaly detection.
        """
        # Check if model is trained
        if self.isolation_forest is None:
            print("Warning: No trained model found. Training new model...")
            self.train_isolation_forest(daily_df)
        
        print("Detecting anomalies with ML models...")
        start_time = time.perf_counter()

        # Create features
        if use_advanced_features:
            features_df = self.create_advanced_features(daily_df)
        elif self.training_mode == "essential":
            features_df = self._create_essential_features(daily_df)
        else:
            # Use basic features
            features_df = daily_df[self.feature_columns].copy()

        expected_features = list(self.feature_columns) if self.feature_columns else list(features_df.columns)
        if hasattr(self.scaler_robust, "feature_names_in_"):
            expected_features = list(self.scaler_robust.feature_names_in_)

        if hasattr(self.scaler_robust, "n_features_in_") and expected_features:
            if len(expected_features) != self.scaler_robust.n_features_in_:
                print("⚠️ Feature mismatch detected. Retraining ML model for consistency...")
                use_essential = self.training_mode == "essential" or not use_advanced_features
                self.train_isolation_forest(daily_df, use_essential_features=use_essential)
                if use_advanced_features:
                    features_df = self.create_advanced_features(daily_df)
                elif self.training_mode == "essential":
                    features_df = self._create_essential_features(daily_df)
                else:
                    features_df = daily_df[self.feature_columns].copy()
                expected_features = list(self.feature_columns) if self.feature_columns else list(features_df.columns)

        # Ensure all expected features are present
        for col in expected_features:
            if col not in features_df.columns:
                features_df[col] = 0

        # Drop unexpected features and enforce order
        features_df = features_df[expected_features]

        features_df = self._sanitize_features(features_df)
        
        # Scale features
        X_scaled = self.scaler_robust.transform(features_df.to_numpy())
        
        # Apply PCA if trained
        if self.pca is not None:
            # Apply PCA transform when available
            X_transformed = self.pca.transform(X_scaled)
        else:
            X_transformed = X_scaled
        
        # Get predictions from Isolation Forest
        predictions = self.isolation_forest.predict(X_transformed)
        scores = self.isolation_forest.score_samples(X_transformed)
        anomaly_probs = self._scores_to_probabilities(scores)
        
        # Create results DataFrame
        base_cols = ['employee_id', 'employee_name', 'date', 'work_duration_hours']

        if 'department' in daily_df.columns:
            base_cols.insert(2, 'department')

        results_df = daily_df[base_cols].copy()
        
        # Add ML predictions
        results_df['ml_anomaly_score'] = anomaly_probs
        results_df['ml_anomaly_flag'] = (anomaly_probs >= probability_threshold).astype(int)
        results_df['ml_prediction'] = predictions
        
        # Add severity levels
        results_df['ml_severity'] = results_df['ml_anomaly_score'].apply(
            self._classify_severity
        )
        
        # Add feature contributions (optional, can be slow on large feature sets)
        if include_contributions and len(daily_df) <= 500:  # Only for small datasets
            contributions = self._get_feature_contributions(features_df, X_transformed)
            results_df['top_contributing_feature'] = contributions['top_features']
            results_df['contribution_score'] = contributions['contribution_scores']
            results_df['feature_contributions'] = contributions['contributions_json']
        else:
            results_df['top_contributing_feature'] = 'N/A'
            results_df['contribution_score'] = 0.0
            results_df['feature_contributions'] = '{}'
        
        # Add confidence metrics
        results_df['confidence_score'] = 1 - (2 * np.abs(anomaly_probs - 0.5))
        
        # Buffer for adaptive learning
        if enable_adaptive:
            self._add_to_learning_buffer(daily_df, results_df)
        
        elapsed_time = time.perf_counter() - start_time
        print(f"✓ ML anomaly detection complete ({elapsed_time:.2f}s)")
        print(f"  - Detected: {results_df['ml_anomaly_flag'].sum()} anomalies")
        print(f"  - Average score: {results_df['ml_anomaly_score'].mean():.3f}")
        
        return results_df
    
    def _get_fallback_results(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Return basic results when ML detection fails."""
        base_cols = ['employee_id', 'employee_name', 'date', 'work_duration_hours']
        if 'department' in daily_df.columns:
            base_cols.insert(2, 'department')
        
        results_df = daily_df[base_cols].copy()
        results_df['ml_anomaly_score'] = 0.5  # Neutral score
        results_df['ml_anomaly_flag'] = 0  # No anomalies detected
        results_df['ml_severity'] = 'Normal'
        results_df['confidence_score'] = 0.0  # Low confidence
        results_df['ml_prediction'] = 1  # Normal prediction
        results_df['top_contributing_feature'] = 'N/A'
        results_df['contribution_score'] = 0.0
        results_df['feature_contributions'] = '{}'
        
        return results_df
    
    def detect_anomalies(self, daily_df: pd.DataFrame, 
                        probability_threshold: float = 0.7,
                        use_advanced_features: bool = True,
                        include_contributions: bool = False,
                        enable_adaptive: bool = True,
                        debug_steps: bool = False,
                        timeout_seconds: Optional[int] = None,
                        sample_size: Optional[int] = 1000,
                        fast_mode: bool = False) -> pd.DataFrame:
        """
        Detect anomalies with timeout protection and optional sampling.
        
        Args:
            daily_df: Daily attendance summaries
            probability_threshold: Threshold for anomaly classification
            use_advanced_features: Whether to use advanced feature engineering
            include_contributions: Whether to include feature contributions
            enable_adaptive: Whether to enable adaptive learning
            debug_steps: Whether to print debug steps
            timeout_seconds: Maximum time allowed for detection (None disables timeout)
            sample_size: Optional sample size for large datasets
            
        Returns:
            DataFrame with anomaly predictions and explanations
        """
        if fast_mode:
            use_advanced_features = False
            sample_size = None

            if self.training_mode != "essential":
                print("Fast ML mode enabled. Retraining with essential features...")
                self.train_isolation_forest(daily_df, use_essential_features=True)

        # Use sampling for very large datasets
        if sample_size and len(daily_df) > sample_size * 1.5:
            print(f"Large dataset ({len(daily_df)} rows). Using sampling ({sample_size} rows)...")
            
            # Stratified sample by employee
            sample_df = daily_df.groupby('employee_id', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, sample_size // daily_df['employee_id'].nunique())))
            ).reset_index(drop=True)
            
            # Run detection on sample with timeout
            try:
                if timeout_seconds:
                    with self._timeout(timeout_seconds):
                        sample_results = self._detect_anomalies_impl(
                            sample_df, probability_threshold, use_advanced_features,
                            include_contributions, False, debug_steps  # Disable adaptive for sample
                        )
                else:
                    sample_results = self._detect_anomalies_impl(
                        sample_df, probability_threshold, use_advanced_features,
                        include_contributions, False, debug_steps
                    )
                
                # Create results for full dataset based on sample
                results_df = self._extend_results(sample_results, daily_df)
                results_df['ml_anomaly_flag'] = 0  # Reset flags for non-sampled data
                results_df['ml_severity'] = 'Normal'
                
                # Keep anomalies only from sample
                anomalies_mask = sample_results['ml_anomaly_flag'] == 1
                if anomalies_mask.any():
                    anomaly_indices = sample_results[anomalies_mask].index
                    for idx in anomaly_indices:
                        if idx < len(results_df):
                            results_df.loc[idx, 'ml_anomaly_flag'] = 1
                            results_df.loc[idx, 'ml_severity'] = sample_results.loc[idx, 'ml_severity']
                
                return results_df
                
            except TimeoutError:
                print(f"⚠️ ML detection timed out after {timeout_seconds}s. Returning fallback results...")
                return self._get_fallback_results(daily_df)
            except Exception as e:
                print(f"⚠️ ML detection failed: {e}. Returning fallback results...")
                return self._get_fallback_results(daily_df)
        
        # Run full detection with timeout
        try:
            if timeout_seconds:
                with self._timeout(timeout_seconds):
                    return self._detect_anomalies_impl(
                        daily_df, probability_threshold, use_advanced_features,
                        include_contributions, enable_adaptive, debug_steps
                    )
            return self._detect_anomalies_impl(
                daily_df, probability_threshold, use_advanced_features,
                include_contributions, enable_adaptive, debug_steps
            )
        except TimeoutError:
            print(f"⚠️ ML detection timed out after {timeout_seconds}s. Returning fallback results...")
            return self._get_fallback_results(daily_df)
        except Exception as e:
            print(f"⚠️ ML detection failed: {e}. Returning fallback results...")
            return self._get_fallback_results(daily_df)
    
    def _extend_results(self, sample_results: pd.DataFrame, 
                       full_daily_df: pd.DataFrame) -> pd.DataFrame:
        """Extend sample results to full dataset."""
        # Create base results for full dataset
        base_cols = ['employee_id', 'employee_name', 'date', 'work_duration_hours']
        if 'department' in full_daily_df.columns:
            base_cols.insert(2, 'department')
        
        results_df = full_daily_df[base_cols].copy()
        
        # Add ML columns with default values
        for col in ['ml_anomaly_score', 'ml_anomaly_flag', 'ml_severity', 
                   'confidence_score', 'ml_prediction', 'top_contributing_feature',
                   'contribution_score', 'feature_contributions']:
            if col in sample_results.columns:
                if col in ['ml_anomaly_score', 'confidence_score', 'contribution_score']:
                    results_df[col] = sample_results[col].mean() if len(sample_results) > 0 else 0.5
                elif col == 'ml_anomaly_flag':
                    results_df[col] = 0
                elif col == 'ml_severity':
                    results_df[col] = 'Normal'
                elif col == 'ml_prediction':
                    results_df[col] = 1
                elif col == 'top_contributing_feature':
                    results_df[col] = 'N/A'
                elif col == 'feature_contributions':
                    results_df[col] = '{}'
        
        return results_df
    
    def _get_feature_contributions(self, features_df: pd.DataFrame, 
                                  X_transformed: np.ndarray) -> Dict[str, Any]:
        """
        Get feature contributions for each prediction.
        
        Args:
            features_df: Original features
            X_transformed: Transformed features (after PCA)
            
        Returns:
            Dictionary with contribution analysis
        """
        top_features = []
        contribution_scores = []
        contributions_json = []
        
        for idx in range(len(features_df)):
            sample_features = features_df.iloc[idx]
            
            if self.pca is not None and hasattr(self.pca, 'components_'):
                # For PCA-transformed features, we need to map back
                sample_transformed = X_transformed[idx]
                
                # Approximate contribution in original space
                # This is simplified - in practice you'd use SHAP or similar
                contributions = np.abs(self.pca.components_.T.dot(sample_transformed))
                
                if contributions.sum() > 0:
                    contributions = contributions / contributions.sum()
            else:
                # For non-PCA, use feature values weighted by importance
                contributions = np.zeros(len(sample_features))
                for i, (feature, importance) in enumerate(self.feature_importance.items()):
                    if i < len(contributions):
                        # Use absolute deviation from median as contribution
                        feature_median = features_df[feature].median()
                        deviation = abs(sample_features[feature] - feature_median)
                        contributions[i] = deviation * importance
            
            # Get top contributing feature
            if len(contributions) > 0:
                top_idx = np.argmax(contributions)
                top_feature = features_df.columns[top_idx]
                top_score = contributions[top_idx]
                
                top_features.append(top_feature)
                contribution_scores.append(top_score)
                
                # Create JSON of top 3 contributions
                top_indices = np.argsort(contributions)[-3:][::-1]
                top_contribs = {
                    features_df.columns[i]: float(contributions[i])
                    for i in top_indices
                }
                contributions_json.append(json.dumps(top_contribs))
            else:
                top_features.append('unknown')
                contribution_scores.append(0.0)
                contributions_json.append('{}')
        
        return {
            'top_features': top_features,
            'contribution_scores': contribution_scores,
            'contributions_json': contributions_json
        }
    
    def _classify_severity(self, anomaly_score: float) -> str:
        """Classify anomaly severity based on ML score."""
        if anomaly_score >= 0.9:
            return 'Critical'
        elif anomaly_score >= 0.75:
            return 'High'
        elif anomaly_score >= 0.6:
            return 'Medium'
        elif anomaly_score >= 0.4:
            return 'Low'
        else:
            return 'Normal'
    
    def _add_to_learning_buffer(self, daily_df: pd.DataFrame, 
                               results_df: pd.DataFrame):
        """Add new predictions to learning buffer for adaptive updates."""
        # Combine data and results
        combined = pd.concat([
            daily_df.reset_index(drop=True),
            results_df[['ml_anomaly_score', 'ml_anomaly_flag']].reset_index(drop=True)
        ], axis=1)
        
        # Keep samples for learning
        learning_samples = combined[
            (combined['ml_anomaly_flag'] == 1) |  # Anomalies
            (combined['ml_anomaly_score'] < 0.2)   # Clear normals
        ].copy()
        
        if len(learning_samples) > 0:
            self.new_data_buffer.append(learning_samples)
            print(f"Added {len(learning_samples)} samples to learning buffer")
            
            # Check if we should update the model
            if self._should_update_model():
                print("Triggering adaptive model update...")
                self.adaptive_update()
    
    def _should_update_model(self) -> bool:
        """Determine if model should be updated."""
        # Check buffer size
        total_buffer_samples = sum(len(batch) for batch in self.new_data_buffer)
        if total_buffer_samples < self.min_samples_retrain:
            return False
        
        # Check time since last retrain
        if self.last_retrain_date:
            days_since_retrain = (datetime.now() - self.last_retrain_date).days
            if days_since_retrain >= self.retrain_interval_days:
                return True
        
        return total_buffer_samples >= (self.min_samples_retrain * 2)
    
    def adaptive_update(self, incremental: bool = True):
        """
        Update model adaptively with new data.
        
        Args:
            incremental: Whether to update incrementally (if possible)
        """
        if not self.new_data_buffer:
            print("No new data for adaptive update")
            return
        
        # Combine buffer data
        new_data = pd.concat(self.new_data_buffer, ignore_index=True).drop_duplicates()
        
        print(f"Performing adaptive update with {len(new_data)} new samples")
        
        if incremental and self.isolation_forest is not None:
            try:
                # Prepare features
                features_df = self.create_advanced_features(new_data)
                X_scaled = self.scaler_robust.transform(features_df)
                
                if self.pca is not None:
                    #X_transformed = self.pca.transform(X_scaled)
                    if getattr(self, "pca", None) is not None:
                        X_scaled = self.pca.transform(X_scaled)
                else:
                    X_transformed = X_scaled
                
                # Update contamination rate based on new data
                self._update_contamination_rate(new_data)
                
                # Partial fit (warm start)
                print("Performing incremental update...")
                # Note: IsolationForest doesn't have partial_fit, so we do warm start
                self._warm_start_update(X_transformed, new_data)
                
            except Exception as e:
                print(f"Incremental update failed: {e}")
                print("Falling back to full retrain...")
                self.train_isolation_forest(new_data)
        else:
            # Full retrain
            self.train_isolation_forest(new_data)
        
        # Clear buffer
        self.new_data_buffer = []
        
        print("✓ Adaptive update complete")
    
    def _update_contamination_rate(self, new_data: pd.DataFrame):
        """Update contamination rate based on new data patterns."""
        if 'ml_anomaly_flag' in new_data.columns:
            new_anomaly_rate = new_data['ml_anomaly_flag'].mean()
            
            # Adaptive update: blend old and new rates
            self.contamination = (1 - self.adaptation_rate) * self.contamination + \
                                self.adaptation_rate * new_anomaly_rate
            
            # Bound between 1% and 30%
            self.contamination = max(0.01, min(0.3, self.contamination))
            
            print(f"Updated contamination rate to {self.contamination:.3f}")
    
    def _warm_start_update(self, X_new: np.ndarray, new_data: pd.DataFrame):
        """Update Isolation Forest with warm start."""
        # Create new Isolation Forest with same parameters
        n_estimators = self.isolation_forest.n_estimators
        
        # Increase estimators for adaptation
        new_n_estimators = int(n_estimators * 1.1)
        
        # Create new forest
        new_forest = IsolationForest(
            n_estimators=new_n_estimators,
            max_samples=self.isolation_forest.max_samples,
            contamination=self.contamination,
            random_state=self.isolation_forest.random_state,
            n_jobs=-1
        )
        
        print("Training updated model...")
        new_forest.fit(X_new)
        
        # Update model
        self.isolation_forest = new_forest
        self.last_retrain_date = datetime.now()
        
        # Save updated model
        self._save_models()
    
    def explain_anomaly(self, sample_idx: int, 
                       daily_df: pd.DataFrame, 
                       results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate detailed explanation for an anomaly.
        
        Args:
            sample_idx: Index of the sample
            daily_df: Original data
            results_df: ML results
            
        Returns:
            Detailed explanation dictionary
        """
        if sample_idx >= len(daily_df):
            return {"error": "Invalid sample index"}
        
        # Get sample data
        sample_data = daily_df.iloc[sample_idx].to_dict()
        sample_results = results_df.iloc[sample_idx].to_dict()
        
        # Create features for this sample
        features_df = self.create_advanced_features(
            pd.DataFrame([sample_data])
        )
        
        explanation = {
            'employee': {
                'id': sample_data.get('employee_id', 'Unknown'),
                'name': sample_data.get('employee_name', 'Unknown'),
                'department': sample_data.get('department', 'Unknown')
            },
            'date': sample_data.get('date', 'Unknown'),
            'anomaly_score': float(sample_results.get('ml_anomaly_score', 0)),
            'severity': sample_results.get('ml_severity', 'Unknown'),
            'is_anomaly': bool(sample_results.get('ml_anomaly_flag', 0)),
            'key_findings': [],
            'feature_analysis': [],
            'recommendations': []
        }
        
        # Analyze top contributing features
        if 'feature_contributions' in sample_results:
            try:
                contribs = json.loads(sample_results['feature_contributions'])
                for feature, score in contribs.items():
                    feature_value = features_df[feature].iloc[0] if feature in features_df.columns else 'N/A'
                    
                    # Get feature statistics
                    feature_stats = self._get_feature_statistics(feature, daily_df)
                    
                    explanation['feature_analysis'].append({
                        'feature': feature,
                        'contribution': float(score),
                        'value': float(feature_value) if isinstance(feature_value, (int, float)) else feature_value,
                        'comparison': feature_stats
                    })
            except:
                pass
        
        # Generate key findings
        explanation['key_findings'] = self._generate_key_findings(
            sample_data, explanation['feature_analysis']
        )
        
        # Generate recommendations
        explanation['recommendations'] = self._generate_recommendations(
            explanation['severity'], explanation['key_findings']
        )
        
        return explanation
    
    def _get_feature_statistics(self, feature: str, daily_df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics for a feature."""
        if feature not in daily_df.columns:
            return {}
        
        feature_data = daily_df[feature].dropna()
        
        if len(feature_data) == 0:
            return {}
        
        return {
            'mean': float(feature_data.mean()),
            'median': float(feature_data.median()),
            'std': float(feature_data.std()),
            'min': float(feature_data.min()),
            'max': float(feature_data.max()),
            'q25': float(feature_data.quantile(0.25)),
            'q75': float(feature_data.quantile(0.75))
        }
    
    def _generate_key_findings(self, sample_data: Dict, 
                              feature_analysis: List[Dict]) -> List[str]:
        """Generate human-readable key findings."""
        findings = []
        
        # Time-based findings
        if 'lateness_minutes' in sample_data and sample_data['lateness_minutes'] > 30:
            findings.append(f"Arrived {sample_data['lateness_minutes']} minutes late")
        
        if 'work_duration_hours' in sample_data:
            hours = sample_data['work_duration_hours']
            if hours < 4:
                findings.append(f"Short work day: {hours:.1f} hours")
            elif hours > 12:
                findings.append(f"Long work day: {hours:.1f} hours")
        
        # Pattern findings
        if 'has_incomplete_pairs' in sample_data and sample_data['has_incomplete_pairs']:
            findings.append("Incomplete attendance pairs detected")
        
        if 'unique_locations' in sample_data and sample_data['unique_locations'] > 1:
            findings.append(f"Multiple locations ({sample_data['unique_locations']}) in one day")
        
        # ML feature findings
        for analysis in feature_analysis[:3]:  # Top 3 features
            feature = analysis['feature']
            value = analysis['value']
            
            if 'deviation' in feature.lower():
                findings.append(f"Unusual pattern in {feature.replace('_', ' ')}")
            elif 'lag' in feature:
                findings.append(f"Significant change from previous days in {feature.replace('_', ' ')}")
        
        return findings
    
    def _generate_recommendations(self, severity: str, 
                                findings: List[str]) -> List[str]:
        """Generate recommendations based on severity and findings."""
        recommendations = []
        
        # Severity-based recommendations
        if severity in ['Critical', 'High']:
            recommendations.append("Immediate review required")
            recommendations.append("Consider discussing with employee")
        
        if severity == 'Medium':
            recommendations.append("Monitor for recurring patterns")
            recommendations.append("Review with supervisor if pattern continues")
        
        # Finding-specific recommendations
        if any('late' in finding.lower() for finding in findings):
            recommendations.append("Check if lateness is justified (traffic, appointments)")
        
        if any('short' in finding.lower() for finding in findings):
            recommendations.append("Verify if employee had approved leave or appointments")
        
        if any('location' in finding.lower() for finding in findings):
            recommendations.append("Confirm work locations with employee")
        
        # General recommendations
        recommendations.append("Document findings in employee record")
        recommendations.append("Schedule follow-up if issues persist")
        
        return recommendations
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report for the ML detector."""
        if not self.performance_history:
            return {"status": "No performance history available"}
        
        latest_perf = self.performance_history[-1]
        
        report = {
            'model_status': 'Trained' if self.isolation_forest else 'Not trained',
            'last_training': self.last_retrain_date.isoformat() if self.last_retrain_date else 'Never',
            'contamination_rate': self.contamination,
            'feature_count': len(self.feature_columns) if self.feature_columns else 0,
            'max_features': self.max_features,
            'performance_history': {
                'total_trainings': len(self.performance_history),
                'latest_training': latest_perf
            },
            'top_features': list(self.feature_importance.items())[:10],
            'learning_buffer': {
                'batches': len(self.new_data_buffer),
                'total_samples': sum(len(batch) for batch in self.new_data_buffer)
            }
        }
        
        return report
    
    def retrain_on_demand(self, daily_df: pd.DataFrame, 
                         force_retrain: bool = False) -> Dict[str, Any]:
        """
        Manual retraining of the ML model.
        
        Args:
            daily_df: New data for training
            force_retrain: Whether to force retrain even if recent
            
        Returns:
            Training results
        """
        if not force_retrain and self.last_retrain_date:
            days_since = (datetime.now() - self.last_retrain_date).days
            if days_since < self.retrain_interval_days:
                return {
                    "status": "skipped",
                    "message": f"Last retrained {days_since} days ago",
                    "next_retrain_in": self.retrain_interval_days - days_since
                }
        
        print("Manual retraining initiated...")
        results = self.train_isolation_forest(daily_df)
        
        return {
            "status": "success",
            "results": results,
            "message": "Model retrained successfully"
        }


# Factory function for easy usage
def create_ml_detector(model_dir: str = 'models/ml_detector',
                      contamination: float = 0.1,
                      adaptation_rate: float = 0.2,
                      max_features: int = 50) -> MLAnomalyDetector:
    """
    Factory function to create and initialize ML detector.
    
    Args:
        model_dir: Directory for model storage
        contamination: Expected anomaly rate
        adaptation_rate: Learning rate for adaptation
        max_features: Maximum number of features to use
        
    Returns:
        Initialized MLAnomalyDetector instance
    """
    detector = MLAnomalyDetector(
        model_dir=model_dir,
        contamination=contamination,
        adaptation_rate=adaptation_rate,
        max_features=max_features
    )
    
    return detector


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("ML Anomaly Detector - Isolation Forest Based")
    print("=" * 60)
    print("\nFeatures:")
    print("1. Primary: Adaptive Isolation Forest")
    print("2. Optimized feature engineering for large datasets")
    print("3. Automatic model persistence")
    print("4. Adaptive learning from new data")
    print("5. Timeout protection and fallback results")
    print("6. Performance monitoring")
    
    print("\nOptimizations:")
    print("- Essential features for large datasets (>1000 rows)")
    print("- Feature limit: 50 by default")
    print("- Timeout protection: 30 seconds")
    print("- Sampling for very large datasets")
    
    print("\nQuick Start:")
    print("from anomaly_detection.ml_detector import create_ml_detector")
    print("detector = create_ml_detector()")
    print("detector.train_isolation_forest(daily_data)")
    print("anomalies = detector.detect_anomalies(new_data, timeout_seconds=30)")
    
    print("\nAdaptive Learning:")
    print("detector.adaptive_update()  # Manual trigger")
    print("detector.retrain_on_demand(new_data)  # Manual retrain")
    
    print("\nMonitoring:")
    print("report = detector.get_performance_report()")
