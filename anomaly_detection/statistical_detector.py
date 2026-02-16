# anomaly_detection/statistical_detector.py
"""
Statistical Anomaly Detection Module
Uses statistical methods to detect anomalies in attendance data.
Adaptive system that can be retrained with new data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import os


class AdaptiveStatisticalDetector:
    """
    Adaptive statistical anomaly detection system.
    
    Features:
    1. Multiple statistical methods
    2. Adaptive learning from new data
    3. Ensemble scoring
    4. Model persistence
    5. Performance monitoring
    """
    
    def __init__(self, 
                 model_dir: str = 'models/statistical',
                 contamination: float = 0.1,
                 adaptation_rate: float = 0.1,
                 retrain_threshold: int = 100):
        """
        Initialize the statistical detector.
        
        Parameters:
        -----------
        model_dir : str
            Directory to save/load models
        contamination : float
            Expected proportion of outliers in the data (0.0 to 0.5)
        adaptation_rate : float
            Rate at which new data influences the model (0.0 to 1.0)
        retrain_threshold : int
            Number of new samples before triggering retraining
        """
        self.model_dir = model_dir
        self.contamination = contamination
        self.adaptation_rate = adaptation_rate
        self.retrain_threshold = retrain_threshold
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize detectors
        self.detectors = {
            'isolation_forest': None,
            'one_class_svm': None,
            'local_outlier_factor': None
        }
        
        # Feature scalers
        self.scaler_standard = StandardScaler()
        self.scaler_robust = RobustScaler()
        
        # Feature importance
        self.feature_importance = {}
        
        # Performance tracking
        self.performance_history = []
        self.new_data_buffer = []
        self.samples_since_retrain = 0
        
        # Feature columns for training
        self.feature_columns = None
        
        # Load existing models if available
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load existing trained models from disk"""
        for model_name in self.detectors.keys():
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            if os.path.exists(model_path):
                try:
                    self.detectors[model_name] = joblib.load(model_path)
                    print(f"Loaded {model_name} from {model_path}")
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        for model_name, model in self.detectors.items():
            if model is not None:
                model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
                joblib.dump(model, model_path)
                print(f"Saved {model_name} to {model_path}")
    
    def prepare_features(self, daily_df: pd.DataFrame, 
                        training_mode: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for statistical anomaly detection.
        
        Args:
            daily_df: Daily attendance summaries
            training_mode: Whether this is for training (include all features)
            
        Returns:
            Tuple of (features DataFrame, feature names)
        """
        df = daily_df.copy()
        
        # Select features for anomaly detection
        feature_candidates = [
            # Time-based features
            'work_duration_hours',
            'lateness_minutes',
            'early_departure_minutes',
            'arrival_total_minutes',
            'departure_total_minutes',
            'overtime_hours',
            
            # Pattern-based features
            'total_checks',
            'check_ins_count',
            'check_outs_count',
            'sequence_issues_count',
            
            # Location-based features
            'unique_locations',
            'location_changes',
            
            # Statistical features
            'arrival_z_score',
            'hours_z_score',
            
            # Day features
            'day_of_week',
            'is_weekend',
            'is_holiday'
        ]
        
        # Only use available features
        available_features = [f for f in feature_candidates if f in df.columns]
        
        # Add derived features
        if 'arrival_total_minutes' in df.columns:
            df['arrival_hour_sin'] = np.sin(2 * np.pi * df['arrival_total_minutes'] / (24 * 60))
            df['arrival_hour_cos'] = np.cos(2 * np.pi * df['arrival_total_minutes'] / (24 * 60))
            available_features.extend(['arrival_hour_sin', 'arrival_hour_cos'])
        
        if 'departure_total_minutes' in df.columns:
            df['departure_hour_sin'] = np.sin(2 * np.pi * df['departure_total_minutes'] / (24 * 60))
            df['departure_hour_cos'] = np.cos(2 * np.pi * df['departure_total_minutes'] / (24 * 60))
            available_features.extend(['departure_hour_sin', 'departure_hour_cos'])
        
        # Add interaction features
        if all(col in df.columns for col in ['work_duration_hours', 'total_checks']):
            df['checks_per_hour'] = df['total_checks'] / (df['work_duration_hours'] + 1e-6)
            available_features.append('checks_per_hour')
        
        if all(col in df.columns for col in ['work_duration_hours', 'lateness_minutes']):
            df['productivity_score'] = df['work_duration_hours'] / (df['lateness_minutes'] + 1)
            available_features.append('productivity_score')
        
        # Handle missing values
        for col in available_features:
            if df[col].isnull().any():
                if df[col].dtype in ['int64', 'float64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
        
        # Store feature columns if in training mode
        if training_mode and not self.feature_columns:
            self.feature_columns = available_features
        
        # Use stored feature columns if available (for consistency)
        if self.feature_columns:
            # Ensure all stored features exist
            features_to_use = [f for f in self.feature_columns if f in df.columns]
            # Add missing features with zeros
            for f in self.feature_columns:
                if f not in df.columns:
                    df[f] = 0
            features_to_use = self.feature_columns
        else:
            features_to_use = available_features
        
        # ---- Ensure all features are numeric (RobustScaler/means require numeric) ----
        for col in features_to_use:
            # Convert booleans to 0/1
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)

            # If it's object but looks numeric, coerce
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Fill any remaining NaNs after coercion
        for col in features_to_use:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
        
        # Create features DataFrame
        features_df = df[features_to_use].copy()
        
        return features_df, features_to_use
    
    def train_models(self, daily_df: pd.DataFrame, 
                    features_df: Optional[pd.DataFrame] = None,
                    feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train statistical anomaly detection models.
        
        Args:
            daily_df: Daily attendance summaries
            features_df: Pre-computed features (optional)
            feature_names: List of feature names (optional)
            
        Returns:
            Dictionary with training results
        """
        print("Training statistical anomaly detection models...")
        
        # Prepare features if not provided
        if features_df is None or feature_names is None:
            features_df, feature_names = self.prepare_features(daily_df, training_mode=True)
        
        # Scale features
        X_scaled_standard = self.scaler_standard.fit_transform(features_df)
        X_scaled_robust = self.scaler_robust.fit_transform(features_df)
        
        # Use robust scaling for better outlier resistance
        X_train = X_scaled_robust
        
        training_results = {}
        
        # 1. Isolation Forest
        print("Training Isolation Forest...")
        iso_forest = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(X_train)
        self.detectors['isolation_forest'] = iso_forest
        
        # Calculate feature importance for Isolation Forest
        # IsolationForest in sklearn does not expose feature_importances_.
        # Use a simple proxy importance based on feature dispersion (robust, always available).
        X_train_df = pd.DataFrame(X_train, columns=feature_names)

        proxy_importance = X_train_df.std().replace(0, 1e-6)
        proxy_importance = (proxy_importance / proxy_importance.sum()).values

        self.feature_importance['isolation_forest'] = dict(zip(feature_names, proxy_importance))

        # feature_importance = np.abs(iso_forest.feature_importances_)
        # self.feature_importance['isolation_forest'] = dict(zip(feature_names, feature_importance))
        
        training_results['isolation_forest'] = {
            'n_features': len(feature_names),
            'contamination': self.contamination
        }
        
        # 2. One-Class SVM (for large datasets)
        if len(daily_df) > 100:
            print("Training One-Class SVM...")
            try:
                oc_svm = OneClassSVM(
                    nu=self.contamination,
                    kernel='rbf',
                    gamma='scale'
                )
                oc_svm.fit(X_train)
                self.detectors['one_class_svm'] = oc_svm
                training_results['one_class_svm'] = {'trained': True}
            except Exception as e:
                print(f"One-Class SVM training failed: {e}")
                training_results['one_class_svm'] = {'trained': False, 'error': str(e)}
        
        # 3. Local Outlier Factor
        print("Training Local Outlier Factor...")
        try:
            lof = LocalOutlierFactor(
                n_neighbors=min(20, len(daily_df) // 2),
                contamination=self.contamination,
                n_jobs=-1,
                novelty=True  # Enable predict on new data
            )
            lof.fit(X_train)
            self.detectors['local_outlier_factor'] = lof
            training_results['local_outlier_factor'] = {'trained': True}
        except Exception as e:
            print(f"LOF training failed: {e}")
            training_results['local_outlier_factor'] = {'trained': False, 'error': str(e)}
        
        # 4. DBSCAN clustering (for reference)
        print("Performing DBSCAN clustering...")
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            clusters = dbscan.fit_predict(X_train)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            
            training_results['dbscan'] = {
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_percentage': n_noise / len(clusters) * 100
            }
        except Exception as e:
            training_results['dbscan'] = {'error': str(e)}
        
        # 5. Dimensionality analysis with PCA
        print("Performing PCA analysis...")
        try:
            pca = PCA(n_components=min(5, len(feature_names)))
            X_pca = pca.fit_transform(X_train)
            explained_variance = pca.explained_variance_ratio_.sum()
            
            training_results['pca'] = {
                'n_components': pca.n_components_,
                'explained_variance': explained_variance,
                'components': pca.components_.tolist()
            }
        except Exception as e:
            training_results['pca'] = {'error': str(e)}
        
        # Save models
        self._save_models()
        
        # Record performance
        performance_record = {
            'timestamp': datetime.now(),
            'n_samples': len(daily_df),
            'n_features': len(feature_names),
            'contamination': self.contamination,
            'training_results': training_results
        }
        self.performance_history.append(performance_record)
        
        print("Model training complete!")
        return training_results
    
    def detect_anomalies(self, daily_df: pd.DataFrame, 
                        use_ensemble: bool = True,
                        threshold: float = 0.5,
                        z_threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies using statistical methods.
        
        Args:
            daily_df: Daily attendance summaries
            use_ensemble: Whether to use ensemble voting
            threshold: Threshold for anomaly score (0.0 to 1.0)
            
        Returns:
            DataFrame with anomaly scores and predictions
        """
        # Check if models are trained
        if not any(model is not None for model in self.detectors.values()):
            print("Warning: No trained models found. Training new models...")
            self.train_models(daily_df)
        
        # Prepare features
        features_df, feature_names = self.prepare_features(daily_df)
        
        # Ensure scaler is fitted
        if not hasattr(self.scaler_robust, "center_"):
            self.scaler_robust.fit(features_df)

        # Scale features
        X_scaled = self.scaler_robust.transform(features_df)
        
        # Initialize results DataFrame
        base_cols = ['employee_id', 'employee_name', 'date', 'work_duration_hours']

        if 'department' in daily_df.columns:
            base_cols.insert(2, 'department')

        results_df = daily_df[base_cols].copy()

        # results_df = daily_df[['employee_id', 'employee_name', 'department', 
        #                       'date', 'work_duration_hours']].copy()
        
        # Dictionary to store predictions from each model
        predictions = {}
        anomaly_scores = {}
        
        # 1. Isolation Forest predictions
        if self.detectors['isolation_forest'] is not None:
            iso_scores = self.detectors['isolation_forest'].score_samples(X_scaled)
            iso_predictions = self.detectors['isolation_forest'].predict(X_scaled)
            
            # Convert to anomaly scores (higher = more anomalous)
            iso_anomaly_scores = 1 - (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-6)
            predictions['isolation_forest'] = (iso_predictions == -1).astype(int)
            anomaly_scores['isolation_forest'] = iso_anomaly_scores
            
            results_df['iso_forest_score'] = iso_anomaly_scores
            results_df['iso_forest_anomaly'] = (iso_predictions == -1)
        
        # 2. One-Class SVM predictions
        if self.detectors['one_class_svm'] is not None:
            try:
                svm_predictions = self.detectors['one_class_svm'].predict(X_scaled)
                svm_distances = self.detectors['one_class_svm'].decision_function(X_scaled)
                
                # Convert distances to anomaly scores
                svm_anomaly_scores = 1 / (1 + np.exp(-svm_distances))  # Sigmoid
                predictions['one_class_svm'] = (svm_predictions == -1).astype(int)
                anomaly_scores['one_class_svm'] = svm_anomaly_scores
                
                results_df['svm_score'] = svm_anomaly_scores
                results_df['svm_anomaly'] = (svm_predictions == -1)
            except Exception as e:
                print(f"One-Class SVM prediction failed: {e}")
        
        # 3. Local Outlier Factor predictions
        if self.detectors['local_outlier_factor'] is not None:
            try:
                lof_predictions = self.detectors['local_outlier_factor'].predict(X_scaled)
                lof_scores = self.detectors['local_outlier_factor'].score_samples(X_scaled)
                
                # Convert to anomaly scores
                lof_anomaly_scores = 1 - (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-6)
                predictions['local_outlier_factor'] = (lof_predictions == -1).astype(int)
                anomaly_scores['local_outlier_factor'] = lof_anomaly_scores
                
                results_df['lof_score'] = lof_anomaly_scores
                results_df['lof_anomaly'] = (lof_predictions == -1)
            except Exception as e:
                print(f"LOF prediction failed: {e}")
        
        # 4. Statistical Z-score method
        print("Applying statistical Z-score method...")
        statistical_anomalies = self._apply_statistical_methods(features_df, feature_names, z_threshold=z_threshold)
        predictions['statistical'] = statistical_anomalies['anomaly_flags']
        anomaly_scores['statistical'] = statistical_anomalies['anomaly_scores']
        
        results_df['statistical_score'] = statistical_anomalies['anomaly_scores']
        results_df['statistical_anomaly'] = statistical_anomalies['anomaly_flags']
        
        # 5. Ensemble voting
        if use_ensemble and len(predictions) > 1:
            ensemble_results = self._ensemble_voting(predictions, anomaly_scores, threshold)
            
            results_df['ensemble_score'] = ensemble_results['ensemble_scores']
            results_df['ensemble_anomaly'] = ensemble_results['ensemble_predictions']
            results_df['agreement_score'] = ensemble_results['agreement_scores']
            
            # Feature contributions
            if self.detectors['isolation_forest'] is not None:
                feature_contributions = self._analyze_feature_contributions(
                    features_df, feature_names, self.detectors['isolation_forest']
                )
                results_df['top_contributing_feature'] = feature_contributions['top_features']
                results_df['contribution_score'] = feature_contributions['contribution_scores']
        
        # 6. Severity classification
        if 'ensemble_score' in results_df.columns:
            results_df['severity_level'] = results_df['ensemble_score'].apply(
                self._classify_severity
            )
        
        # 7. Add buffer for adaptive learning
        self._add_to_buffer(daily_df, results_df)
        
        # Check if retraining is needed
        if self.samples_since_retrain >= self.retrain_threshold:
            print(f"Retraining threshold reached ({self.samples_since_retrain} samples). Retraining...")
            self.adaptive_retrain()
        
        return results_df
    
    def _apply_statistical_methods(self, features_df: pd.DataFrame, 
                                  feature_names: List[str],
                                  z_threshold: float = 3.0) -> Dict[str, np.ndarray]:
        """
        Apply traditional statistical methods for anomaly detection.
        """
        anomaly_flags = np.zeros(len(features_df))
        anomaly_scores = np.zeros(len(features_df))
        
        # Z-score method for each feature
        for i, feature in enumerate(feature_names):
            if features_df[feature].std() > 0:
                z_scores = np.abs((features_df[feature] - features_df[feature].mean()) / 
                                 features_df[feature].std())
                
                # Flag anomalies (|Z| > threshold)
                feature_anomalies = (z_scores > z_threshold).astype(int)
                anomaly_flags = np.maximum(anomaly_flags, feature_anomalies)
                
                # Accumulate scores
                anomaly_scores += z_scores / len(feature_names)
        
        # Interquartile Range (IQR) method
        for feature in feature_names:
            Q1 = features_df[feature].quantile(0.25)
            Q3 = features_df[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                iqr_anomalies = ((features_df[feature] < lower_bound) | 
                                (features_df[feature] > upper_bound)).astype(int)
                anomaly_flags = np.maximum(anomaly_flags, iqr_anomalies)
        
        # Normalize scores to [0, 1]
        if anomaly_scores.max() > anomaly_scores.min():
            anomaly_scores = (anomaly_scores - anomaly_scores.min()) / \
                           (anomaly_scores.max() - anomaly_scores.min())
        
        return {
            'anomaly_flags': anomaly_flags,
            'anomaly_scores': anomaly_scores
        }
    
    def _ensemble_voting(self, predictions: Dict[str, np.ndarray], 
                        scores: Dict[str, np.ndarray], 
                        threshold: float) -> Dict[str, np.ndarray]:
        """
        Combine predictions from multiple models using ensemble voting.
        """
        model_names = list(predictions.keys())
        
        # 1. Majority voting
        all_predictions = np.array([predictions[name] for name in model_names])
        vote_counts = all_predictions.sum(axis=0)
        majority_votes = (vote_counts > len(model_names) / 2).astype(int)
        
        # 2. Weighted voting by model performance (simplified)
        ensemble_scores = np.zeros(len(all_predictions[0]))
        for name in model_names:
            ensemble_scores += scores[name]
        ensemble_scores /= len(model_names)
        
        # 3. Threshold-based prediction
        ensemble_predictions = (ensemble_scores > threshold).astype(int)
        
        # 4. Agreement score (how many models agree)
        agreement_scores = vote_counts / len(model_names)
        
        return {
            'ensemble_scores': ensemble_scores,
            'ensemble_predictions': ensemble_predictions,
            'majority_votes': majority_votes,
            'agreement_scores': agreement_scores
        }
    
    def _analyze_feature_contributions(self, features_df: pd.DataFrame, 
                                      feature_names: List[str], 
                                      model: Any) -> Dict[str, List]:
        """
        Analyze which features contribute most to anomaly detection.
        """
        importances = None
        if 'isolation_forest' in self.feature_importance:
            importances = self.feature_importance['isolation_forest']

        if importances is None and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
            # Get top contributing features for each sample
            top_features = []
            contribution_scores = []
            
            for idx in range(len(features_df)):
                # Get feature values for this sample
                sample_values = features_df.iloc[idx].values
                
                # Weight by importance and deviation from mean
                deviations = np.abs(sample_values - features_df.mean().values)
                contributions = deviations * importances
                
                # Get top contributing feature
                top_idx = np.argmax(contributions)
                top_features.append(feature_names[top_idx])
                contribution_scores.append(contributions[top_idx])
            
            return {
                'top_features': top_features,
                'contribution_scores': contribution_scores,
                'feature_importances': dict(zip(feature_names, importances))
            }
        
        # Fallback: use feature with highest deviation
        top_features = []
        contribution_scores = []
        
        for idx in range(len(features_df)):
            deviations = np.abs(features_df.iloc[idx].values - features_df.mean().values)
            top_idx = np.argmax(deviations)
            top_features.append(feature_names[top_idx])
            contribution_scores.append(deviations[top_idx])
        
        return {
            'top_features': top_features,
            'contribution_scores': contribution_scores
        }
    
    def _classify_severity(self, anomaly_score: float) -> str:
        """Classify anomaly severity based on score."""
        if anomaly_score >= 0.8:
            return 'Critical'
        elif anomaly_score >= 0.6:
            return 'High'
        elif anomaly_score >= 0.4:
            return 'Medium'
        elif anomaly_score >= 0.2:
            return 'Low'
        else:
            return 'Normal'
    
    def _add_to_buffer(self, daily_df: pd.DataFrame, results_df: pd.DataFrame):
        """Add new data to buffer for adaptive learning."""
        # Combine data and results
        combined = pd.concat([
            daily_df.reset_index(drop=True),
            results_df[['ensemble_score', 'ensemble_anomaly']].reset_index(drop=True)
        ], axis=1)
        
        # Keep only anomalous or high-scoring normal samples
        buffer_samples = combined[
            (combined['ensemble_anomaly'] == 1) | 
            (combined['ensemble_score'] > 0.7) |
            (combined['ensemble_score'] < 0.1)
        ].copy()
        
        self.new_data_buffer.append(buffer_samples)
        self.samples_since_retrain += len(buffer_samples)
        
        # Keep buffer size manageable
        if len(self.new_data_buffer) > 10:
            self.new_data_buffer = self.new_data_buffer[-10:]
    
    def adaptive_retrain(self, incremental: bool = True):
        """
        Retrain models adaptively with new data.
        
        Args:
            incremental: Whether to update models incrementally
        """
        if not self.new_data_buffer:
            print("No new data for adaptive retraining.")
            return
        
        # Combine buffer data
        new_data = pd.concat(self.new_data_buffer, ignore_index=True).drop_duplicates()
        
        print(f"Adaptive retraining with {len(new_data)} new samples...")
        
        if incremental and self.detectors['isolation_forest'] is not None:
            # Incremental update for Isolation Forest
            try:
                # Prepare features for new data
                features_df, _ = self.prepare_features(new_data)
                X_new = self.scaler_robust.transform(features_df)
                
                # Partial fit (if supported)
                if hasattr(self.detectors['isolation_forest'], 'partial_fit'):
                    self.detectors['isolation_forest'].partial_fit(X_new)
                    print("Incremental update completed for Isolation Forest.")
                else:
                    # Full retrain with combined data
                    print("Performing full retrain...")
                    # In practice, you would combine with historical data
                    # For simplicity, we'll do a full retrain on new data
                    self.train_models(new_data)
                
                # Update adaptation parameters
                self._update_adaptation_parameters(new_data)
                
            except Exception as e:
                print(f"Incremental update failed: {e}. Performing full retrain.")
                self.train_models(new_data)
        else:
            # Full retrain
            self.train_models(new_data)
        
        # Reset counters
        self.new_data_buffer = []
        self.samples_since_retrain = 0
        
        print("Adaptive retraining complete.")
    
    def _update_adaptation_parameters(self, new_data: pd.DataFrame):
        """Update adaptation parameters based on new data."""
        # Adjust contamination rate based on anomaly prevalence
        if 'ensemble_anomaly' in new_data.columns:
            anomaly_rate = new_data['ensemble_anomaly'].mean()
            
            # Smooth update of contamination parameter
            self.contamination = (1 - self.adaptation_rate) * self.contamination + \
                                self.adaptation_rate * anomaly_rate
            
            # Keep within reasonable bounds
            self.contamination = max(0.01, min(0.3, self.contamination))
            
            print(f"Updated contamination rate to {self.contamination:.3f}")
    
    def get_performance_metrics(self) -> pd.DataFrame:
        """Get performance metrics history."""
        if not self.performance_history:
            return pd.DataFrame()
        
        metrics_df = pd.DataFrame(self.performance_history)
        return metrics_df
    
    def get_feature_importance(self) -> Dict[str, Dict]:
        """Get feature importance from trained models."""
        return self.feature_importance
    
    def explain_anomaly(self, sample_idx: int, daily_df: pd.DataFrame, 
                       results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate explanation for a specific anomaly.
        
        Args:
            sample_idx: Index of the sample to explain
            daily_df: Original daily data
            results_df: Anomaly detection results
            
        Returns:
            Dictionary with anomaly explanation
        """
        if sample_idx >= len(daily_df):
            return {"error": "Invalid sample index"}
        
        # Get the sample data
        sample_data = daily_df.iloc[sample_idx].to_dict()
        sample_results = results_df.iloc[sample_idx].to_dict()
        
        # Prepare features for this sample
        features_df, feature_names = self.prepare_features(
            pd.DataFrame([sample_data])
        )
        
        explanation = {
            'employee_id': sample_data.get('employee_id', 'Unknown'),
            'date': sample_data.get('date', 'Unknown'),
            'anomaly_score': sample_results.get('ensemble_score', 0),
            'severity': sample_results.get('severity_level', 'Unknown'),
            'is_anomaly': bool(sample_results.get('ensemble_anomaly', 0)),
            'contributing_factors': [],
            'comparison_to_norm': {}
        }
        
        # Analyze contributing features
        if self.detectors['isolation_forest'] is not None:
            # Get feature importances
            if 'isolation_forest' in self.feature_importance:
                importances = self.feature_importance['isolation_forest']
                
                # Get top 3 contributing features
                for feature in feature_names:
                    if feature in importances:
                        feature_value = features_df[feature].iloc[0]
                        
                        # Calculate deviation from normal
                        # (In practice, you'd compare with historical data)
                        deviation = 0  # Placeholder
                        
                        explanation['contributing_factors'].append({
                            'feature': feature,
                            'importance': importances[feature],
                            'value': feature_value,
                            'deviation': deviation
                        })
                
                # Sort by importance
                explanation['contributing_factors'].sort(
                    key=lambda x: x['importance'], reverse=True
                )
                explanation['contributing_factors'] = explanation['contributing_factors'][:5]
        
        # Model agreement
        model_agreement = {}
        for model in ['iso_forest_anomaly', 'svm_anomaly', 'lof_anomaly', 'statistical_anomaly']:
            if model in sample_results:
                model_agreement[model] = bool(sample_results[model])
        
        explanation['model_agreement'] = model_agreement
        
        return explanation


# Helper function for quick usage
def detect_statistical_anomalies(daily_df: pd.DataFrame, 
                                train_new: bool = False,
                                config: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function for statistical anomaly detection.
    
    Args:
        daily_df: Daily attendance summaries
        train_new: Whether to train new models
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (anomalies_df, training_info)
    """
    if config is None:
        config = {
            'contamination': 0.1,
            'adaptation_rate': 0.1,
            'retrain_threshold': 100
        }
    
    detector = AdaptiveStatisticalDetector(**config)
    
    if train_new or not any(detector.detectors.values()):
        training_info = detector.train_models(daily_df)
    else:
        training_info = {"using_existing_models": True}
    
    anomalies_df = detector.detect_anomalies(daily_df)
    
    return anomalies_df, training_info


if __name__ == "__main__":
    # Example usage
    print("Adaptive Statistical Anomaly Detector")
    print("=" * 50)
    print("Features:")
    print("1. Multiple statistical methods (Isolation Forest, One-Class SVM, LOF)")
    print("2. Ensemble voting for robust detection")
    print("3. Adaptive learning from new data")
    print("4. Feature importance analysis")
    print("5. Model persistence and incremental updates")
    print("\nUsage:")
    print("from anomaly_detection.statistical_detector import AdaptiveStatisticalDetector")
    print("detector = AdaptiveStatisticalDetector()")
    print("anomalies = detector.detect_anomalies(daily_data)")
    print("\nFor adaptive retraining:")
    print("detector.adaptive_retrain()")
