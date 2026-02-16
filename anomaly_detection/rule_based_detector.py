# anomaly_detection/rule_based_detector.py
"""
Rule-Based Anomaly Detection Module
Detects attendance anomalies based on configurable business rules and thresholds
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class RuleBasedAnomalyDetector:
    """
    Rule-based anomaly detection system for attendance data.
    
    This class applies configurable business rules to detect:
    1. Time-based anomalies (lateness, early departure, short/long days)
    2. Pattern-based anomalies (missing pairs, sequence issues)
    3. Location-based anomalies (multiple locations, frequent changes)
    4. Statistical anomalies (deviations from personal patterns)
    5. Behavioral anomalies (weekend/holiday work, consecutive issues)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the detector with configurable rules.
        
        Parameters:
        -----------
        config : Dict, optional
            Dictionary containing rule configurations. If None, uses defaults.
        """
        # Default configuration for Nigerian workplace
        self.default_config = {
            # Time-based rules
            'lateness_threshold_minutes': 30,  # More than 30 minutes late
            'early_departure_threshold_minutes': 60,  # More than 60 minutes early
            'short_day_threshold_hours': 4,  # Less than 4 hours
            'long_day_threshold_hours': 12,  # More than 12 hours
            'very_long_day_threshold_hours': 14,  # More than 14 hours
            'overtime_threshold_hours': 9,  # More than 9 hours (overtime)
            
            # Pattern-based rules
            'min_checks_per_day': 2,  # At least 2 checks (IN and OUT)
            'max_checks_per_day': 8,  # More than 8 checks is excessive
            'consecutive_late_days': 3,  # Late for 3+ consecutive days
            'consecutive_absent_days': 3,  # Absent for 3+ consecutive days
            
            # Location-based rules
            'max_locations_per_day': 2,  # More than 2 locations in a day
            'max_location_changes': 3,  # More than 3 location changes
            
            # Statistical rules
            'z_score_threshold': 2.5,  # |Z| > 2.5 is statistical anomaly
            'rolling_deviation_threshold': 2.0,  # Hours deviating from 7-day avg
            
            # Behavioral rules
            'allow_weekend_work': False,  # Flag weekend work as anomaly
            'allow_holiday_work': False,  # Flag holiday work as anomaly
            'max_consecutive_weekend_work': 2,  # More than 2 weekends in a row
            
            # Severity scoring
            'severity_weights': {
                'time_based': 1.0,
                'pattern_based': 0.8,
                'location_based': 1.2,
                'statistical': 1.5,
                'behavioral': 1.0
            }
        }
        
        # Use defaults, then override with provided config
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # Initialize anomaly categories
        self.anomaly_categories = {
            'time_based': [],
            'pattern_based': [],
            'location_based': [],
            'statistical': [],
            'behavioral': []
        }
        
    def detect_anomalies(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to detect anomalies in daily attendance summaries.
        
        Args:
            daily_df: DataFrame from data_preparation.py with features
            
        Returns:
            DataFrame with detected anomalies and severity scores
        """
        if daily_df.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying original
        df = daily_df.copy()
        
        print(f"Starting anomaly detection for {len(df)} daily records...")
        
        # Step 1: Apply individual detection rules
        anomalies_list = []
        
        for idx, row in df.iterrows():
            day_anomalies = self._detect_anomalies_for_day(row)
            
            if day_anomalies:
                anomaly_record = {
                    'employee_id': row['employee_id'],
                    'employee_name': row.get('employee_name', row['employee_id']),
                    'department': row.get('department', 'Unknown'),
                    'job_title': row.get('job_title', 'Unknown'),
                    'date': row['date'],
                    'day_of_week': row.get('day_name', 'Unknown'),
                    'work_duration_hours': row.get('work_duration_hours', 0),
                    'anomalies': day_anomalies,
                    'anomaly_types': self._extract_anomaly_types(day_anomalies),
                    'anomaly_count': len(day_anomalies)
                }
                
                # Calculate severity score
                anomaly_record['severity_score'] = self._calculate_severity_score(day_anomalies)
                anomaly_record['severity_level'] = self._categorize_severity(anomaly_record['severity_score'])
                
                anomalies_list.append(anomaly_record)
        
        if not anomalies_list:
            print("No anomalies detected.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        anomalies_df = pd.DataFrame(anomalies_list)
        
        # Step 2: Add behavioral patterns (consecutive issues)
        anomalies_df = self._add_behavioral_patterns(anomalies_df, df)
        
        # Step 3: Sort by severity and date
        anomalies_df = anomalies_df.sort_values(
            ['severity_score', 'date'], 
            ascending=[False, False]
        ).reset_index(drop=True)
        
        print(f"Detected {len(anomalies_df)} anomaly records across {anomalies_df['employee_id'].nunique()} employees")
        
        return anomalies_df
    
    def _detect_anomalies_for_day(self, row: pd.Series) -> List[Dict]:
        """
        Detect all anomalies for a single day's record.
        """
        anomalies = []
        
        # 1. Time-based anomalies
        time_anomalies = self._detect_time_anomalies(row)
        anomalies.extend(time_anomalies)
        
        # 2. Pattern-based anomalies
        pattern_anomalies = self._detect_pattern_anomalies(row)
        anomalies.extend(pattern_anomalies)
        
        # 3. Location-based anomalies
        location_anomalies = self._detect_location_anomalies(row)
        anomalies.extend(location_anomalies)
        
        # 4. Statistical anomalies
        statistical_anomalies = self._detect_statistical_anomalies(row)
        anomalies.extend(statistical_anomalies)
        
        # 5. Behavioral anomalies
        behavioral_anomalies = self._detect_behavioral_anomalies(row)
        anomalies.extend(behavioral_anomalies)
        
        return anomalies
    
    def _detect_time_anomalies(self, row: pd.Series) -> List[Dict]:
        """Detect time-based anomalies"""
        anomalies = []
        
        # Lateness anomaly
        lateness = row.get('lateness_minutes', 0)
        if lateness > self.config['lateness_threshold_minutes']:
            anomalies.append({
                'type': 'time_based',
                'subtype': 'excessive_lateness',
                'description': f'Arrived {lateness} minutes late (threshold: {self.config["lateness_threshold_minutes"]} mins)',
                'value': lateness,
                'threshold': self.config['lateness_threshold_minutes'],
                'severity': self._get_lateness_severity(lateness)
            })
        
        # Early departure anomaly
        early_departure = row.get('early_departure_minutes', 0)
        if early_departure > self.config['early_departure_threshold_minutes']:
            anomalies.append({
                'type': 'time_based',
                'subtype': 'early_departure',
                'description': f'Left {early_departure} minutes early (threshold: {self.config["early_departure_threshold_minutes"]} mins)',
                'value': early_departure,
                'threshold': self.config['early_departure_threshold_minutes'],
                'severity': self._get_early_departure_severity(early_departure)
            })
        
        # Short work day anomaly
        work_hours = row.get('work_duration_hours', 0)
        if work_hours < self.config['short_day_threshold_hours']:
            anomalies.append({
                'type': 'time_based',
                'subtype': 'short_work_day',
                'description': f'Only worked {work_hours:.1f} hours (threshold: {self.config["short_day_threshold_hours"]} hours)',
                'value': work_hours,
                'threshold': self.config['short_day_threshold_hours'],
                'severity': self._get_short_day_severity(work_hours)
            })
        
        # Long work day anomaly
        if work_hours > self.config['long_day_threshold_hours']:
            anomalies.append({
                'type': 'time_based',
                'subtype': 'long_work_day',
                'description': f'Worked {work_hours:.1f} hours (threshold: {self.config["long_day_threshold_hours"]} hours)',
                'value': work_hours,
                'threshold': self.config['long_day_threshold_hours'],
                'severity': self._get_long_day_severity(work_hours)
            })
        
        # Very long work day (potential data error)
        if work_hours > self.config['very_long_day_threshold_hours']:
            anomalies.append({
                'type': 'time_based',
                'subtype': 'very_long_work_day',
                'description': f'Worked {work_hours:.1f} hours - possible error (threshold: {self.config["very_long_day_threshold_hours"]} hours)',
                'value': work_hours,
                'threshold': self.config['very_long_day_threshold_hours'],
                'severity': 'High'
            })
        
        # Excessive overtime
        overtime = row.get('overtime_hours', 0)
        if overtime > 4:  # More than 4 hours overtime
            anomalies.append({
                'type': 'time_based',
                'subtype': 'excessive_overtime',
                'description': f'{overtime:.1f} hours overtime',
                'value': overtime,
                'threshold': 4,
                'severity': 'Medium'
            })
        
        return anomalies
    
    def _detect_pattern_anomalies(self, row: pd.Series) -> List[Dict]:
        """Detect pattern-based anomalies"""
        anomalies = []
        
        # Too few checks
        total_checks = row.get('total_checks', 0)
        if total_checks < self.config['min_checks_per_day']:
            anomalies.append({
                'type': 'pattern_based',
                'subtype': 'insufficient_checks',
                'description': f'Only {total_checks} check(s) recorded (minimum: {self.config["min_checks_per_day"]})',
                'value': total_checks,
                'threshold': self.config['min_checks_per_day'],
                'severity': 'High' if total_checks == 0 else 'Medium'
            })
        
        # Too many checks (possible system abuse)
        if total_checks > self.config['max_checks_per_day']:
            anomalies.append({
                'type': 'pattern_based',
                'subtype': 'excessive_checks',
                'description': f'{total_checks} checks recorded (maximum: {self.config["max_checks_per_day"]})',
                'value': total_checks,
                'threshold': self.config['max_checks_per_day'],
                'severity': 'Medium'
            })
        
        # Incomplete IN/OUT pairs
        if row.get('has_incomplete_pairs', False):
            anomalies.append({
                'type': 'pattern_based',
                'subtype': 'incomplete_pairs',
                'description': 'Missing IN or OUT pair',
                'value': None,
                'threshold': None,
                'severity': 'Medium'
            })
        
        # Sequence issues
        if row.get('sequence_issue_flag', False):
            issues = row.get('sequence_issues', 'Unknown issue')
            anomalies.append({
                'type': 'pattern_based',
                'subtype': 'sequence_issue',
                'description': f'Check sequence issue: {issues}',
                'value': None,
                'threshold': None,
                'severity': 'Low'
            })
        
        # Odd number of checks (should be even)
        if total_checks > 0 and total_checks % 2 != 0:
            anomalies.append({
                'type': 'pattern_based',
                'subtype': 'odd_check_count',
                'description': f'Odd number of checks: {total_checks} (should be even)',
                'value': total_checks,
                'threshold': None,
                'severity': 'Low'
            })
        
        return anomalies
    
    def _detect_location_anomalies(self, row: pd.Series) -> List[Dict]:
        """Detect location-based anomalies"""
        anomalies = []
        
        # Multiple locations in one day
        unique_locations = row.get('unique_locations', 0)
        if unique_locations > self.config['max_locations_per_day']:
            anomalies.append({
                'type': 'location_based',
                'subtype': 'multiple_locations',
                'description': f'{unique_locations} different locations in one day (maximum: {self.config["max_locations_per_day"]})',
                'value': unique_locations,
                'threshold': self.config['max_locations_per_day'],
                'severity': 'High' if unique_locations > 3 else 'Medium'
            })
        
        # Frequent location changes
        location_changes = row.get('location_changes', 0)
        if location_changes > self.config['max_location_changes']:
            anomalies.append({
                'type': 'location_based',
                'subtype': 'frequent_location_changes',
                'description': f'{location_changes} location changes (maximum: {self.config["max_location_changes"]})',
                'value': location_changes,
                'threshold': self.config['max_location_changes'],
                'severity': 'Medium'
            })
        
        # Location mismatch (if we had expected locations)
        if 'main_location' in row and 'expected_location' in row:
            if row['main_location'] != row['expected_location']:
                anomalies.append({
                    'type': 'location_based',
                    'subtype': 'unexpected_location',
                    'description': f'Worked from {row["main_location"]} instead of {row["expected_location"]}',
                    'value': row['main_location'],
                    'threshold': row['expected_location'],
                    'severity': 'Medium'
                })
        
        return anomalies
    
    def _detect_statistical_anomalies(self, row: pd.Series) -> List[Dict]:
        """Detect statistical anomalies"""
        anomalies = []
        
        # Z-score based anomalies
        arrival_z = abs(row.get('arrival_z_score', 0))
        hours_z = abs(row.get('hours_z_score', 0))
        
        if arrival_z > self.config['z_score_threshold']:
            z_value = row.get('arrival_z_score', 0)
            anomalies.append({
                'type': 'statistical',
                'subtype': 'arrival_outlier',
                'description': f'Arrival time is statistical outlier (Z-score: {z_value:.2f})',
                'value': z_value,
                'threshold': self.config['z_score_threshold'],
                'severity': 'High' if arrival_z > 3 else 'Medium'
            })
        
        if hours_z > self.config['z_score_threshold']:
            z_value = row.get('hours_z_score', 0)
            anomalies.append({
                'type': 'statistical',
                'subtype': 'hours_outlier',
                'description': f'Work hours are statistical outlier (Z-score: {z_value:.2f})',
                'value': z_value,
                'threshold': self.config['z_score_threshold'],
                'severity': 'High' if hours_z > 3 else 'Medium'
            })
        
        # Deviation from rolling average
        if pd.notna(row.get('arrival_rolling_avg_7d')) and pd.notna(row.get('arrival_total_minutes')):
            avg_arrival = row['arrival_rolling_avg_7d']
            actual_arrival = row['arrival_total_minutes']
            
            if avg_arrival > 0:  # Avoid division by zero
                deviation = abs(actual_arrival - avg_arrival) / 60  # Convert to hours
                
                if deviation > self.config['rolling_deviation_threshold']:
                    anomalies.append({
                        'type': 'statistical',
                        'subtype': 'arrival_deviation',
                        'description': f'Arrival deviates {deviation:.1f} hours from 7-day average',
                        'value': deviation,
                        'threshold': self.config['rolling_deviation_threshold'],
                        'severity': 'Medium'
                    })
        
        return anomalies
    
    def _detect_behavioral_anomalies(self, row: pd.Series) -> List[Dict]:
        """Detect behavioral anomalies"""
        anomalies = []
        
        # Weekend work (if not allowed)
        if not self.config['allow_weekend_work'] and row.get('is_weekend', False):
            if row.get('work_duration_hours', 0) > 1:  # More than 1 hour on weekend
                anomalies.append({
                    'type': 'behavioral',
                    'subtype': 'weekend_work',
                    'description': f'Worked {row["work_duration_hours"]:.1f} hours on weekend',
                    'value': row['work_duration_hours'],
                    'threshold': 1,
                    'severity': 'Low'
                })
        
        # Holiday work (if not allowed)
        if not self.config['allow_holiday_work'] and row.get('is_holiday', False):
            if row.get('work_duration_hours', 0) > 1:
                anomalies.append({
                    'type': 'behavioral',
                    'subtype': 'holiday_work',
                    'description': f'Worked {row["work_duration_hours"]:.1f} hours on holiday',
                    'value': row['work_duration_hours'],
                    'threshold': 1,
                    'severity': 'Low'
                })
        
        # Work on both weekend and holiday
        if row.get('is_weekend', False) and row.get('is_holiday', False):
            if row.get('work_duration_hours', 0) > 1:
                anomalies.append({
                    'type': 'behavioral',
                    'subtype': 'weekend_holiday_work',
                    'description': f'Worked {row["work_duration_hours"]:.1f} hours on weekend holiday',
                    'value': row['work_duration_hours'],
                    'threshold': 1,
                    'severity': 'Medium'
                })
        
        return anomalies
    
    def _add_behavioral_patterns(self, anomalies_df: pd.DataFrame, 
                                daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pattern-based anomalies that require multiple days of data
        """
        if anomalies_df.empty:
            return anomalies_df
        
        # Group by employee for pattern analysis
        employee_anomalies = []
        
        for emp_id in anomalies_df['employee_id'].unique():
            emp_anomalies = anomalies_df[anomalies_df['employee_id'] == emp_id]
            emp_daily = daily_df[daily_df['employee_id'] == emp_id]
            
            if len(emp_anomalies) < 2:
                continue
            
            # Sort by date
            emp_anomalies = emp_anomalies.sort_values('date')
            emp_daily = emp_daily.sort_values('date')
            
            # Detect consecutive late days
            late_patterns = self._detect_consecutive_late_days(emp_daily)
            
            # Detect consecutive absent days (zero work hours)
            absent_patterns = self._detect_consecutive_absent_days(emp_daily)
            
            # Add pattern anomalies to the list
            for pattern in late_patterns + absent_patterns:
                # Find the last day of the pattern in anomalies
                pattern_end_date = pattern['end_date']
                pattern_idx = emp_anomalies[emp_anomalies['date'] == pattern_end_date].index
                
                if not pattern_idx.empty:
                    idx = pattern_idx[0]
                    # Add pattern anomaly to existing record
                    current_anomalies = anomalies_df.at[idx, 'anomalies']
                    current_anomalies.append({
                        'type': 'behavioral',
                        'subtype': pattern['subtype'],
                        'description': pattern['description'],
                        'value': pattern['consecutive_days'],
                        'threshold': pattern['threshold'],
                        'severity': pattern['severity']
                    })
                    
                    # Update anomaly count and types
                    anomalies_df.at[idx, 'anomaly_count'] = len(current_anomalies)
                    anomalies_df.at[idx, 'anomaly_types'] = self._extract_anomaly_types(current_anomalies)
                    
                    # Recalculate severity
                    anomalies_df.at[idx, 'severity_score'] = self._calculate_severity_score(current_anomalies)
                    anomalies_df.at[idx, 'severity_level'] = self._categorize_severity(
                        anomalies_df.at[idx, 'severity_score']
                    )
        
        return anomalies_df
    
    def _detect_consecutive_late_days(self, emp_daily: pd.DataFrame) -> List[Dict]:
        """Detect consecutive late arrival days"""
        patterns = []
        
        if 'is_late' not in emp_daily.columns:
            return patterns
        
        # Find consecutive late days
        consecutive_count = 0
        start_date = None
        
        emp_daily = emp_daily.reset_index(drop=True)
        for idx, row in emp_daily.iterrows():
            if row['is_late']:
                if consecutive_count == 0:
                    start_date = row['date']
                consecutive_count += 1
            else:
                if consecutive_count >= self.config['consecutive_late_days']:
                    patterns.append({
                        'subtype': 'consecutive_late_days',
                        'description': f'Late for {consecutive_count} consecutive days',
                        'consecutive_days': consecutive_count,
                        'start_date': start_date,
                        'end_date': emp_daily.iloc[idx - 1]['date'] if idx > 0 else start_date,
                        'threshold': self.config['consecutive_late_days'],
                        'severity': 'High' if consecutive_count > 5 else 'Medium'
                    })
                consecutive_count = 0
                start_date = None
        
        # Check if pattern continues to the end
        if consecutive_count >= self.config['consecutive_late_days']:
            patterns.append({
                'subtype': 'consecutive_late_days',
                'description': f'Late for {consecutive_count} consecutive days',
                'consecutive_days': consecutive_count,
                'start_date': start_date,
                'end_date': emp_daily.iloc[-1]['date'],
                'threshold': self.config['consecutive_late_days'],
                'severity': 'High' if consecutive_count > 5 else 'Medium'
            })
        
        return patterns
    
    def _detect_consecutive_absent_days(self, emp_daily: pd.DataFrame) -> List[Dict]:
        """Detect consecutive absent days (zero work hours)"""
        patterns = []
        
        if 'work_duration_hours' not in emp_daily.columns:
            return patterns
        
        # Find consecutive absent days
        consecutive_count = 0
        start_date = None
        
        emp_daily = emp_daily.reset_index(drop=True)
        for idx, row in emp_daily.iterrows():
            if row.get('work_duration_hours', 0) == 0:
                if consecutive_count == 0:
                    start_date = row['date']
                consecutive_count += 1
            else:
                if consecutive_count >= self.config['consecutive_absent_days']:
                    patterns.append({
                        'subtype': 'consecutive_absent_days',
                        'description': f'Absent for {consecutive_count} consecutive days',
                        'consecutive_days': consecutive_count,
                        'start_date': start_date,
                        'end_date': emp_daily.iloc[idx - 1]['date'] if idx > 0 else start_date,
                        'threshold': self.config['consecutive_absent_days'],
                        'severity': 'High'
                    })
                consecutive_count = 0
                start_date = None
        
        # Check if pattern continues to the end
        if consecutive_count >= self.config['consecutive_absent_days']:
            patterns.append({
                'subtype': 'consecutive_absent_days',
                'description': f'Absent for {consecutive_count} consecutive days',
                'consecutive_days': consecutive_count,
                'start_date': start_date,
                'end_date': emp_daily.iloc[-1]['date'],
                'threshold': self.config['consecutive_absent_days'],
                'severity': 'High'
            })
        
        return patterns
    
    def _extract_anomaly_types(self, anomalies: List[Dict]) -> str:
        """Extract unique anomaly types as string"""
        types = set()
        for anomaly in anomalies:
            types.add(f"{anomaly['type']}:{anomaly['subtype']}")
        return ', '.join(sorted(types))
    
    def _calculate_severity_score(self, anomalies: List[Dict]) -> float:
        """Calculate composite severity score"""
        if not anomalies:
            return 0.0
        
        score = 0.0
        
        for anomaly in anomalies:
            type_weight = self.config['severity_weights'].get(anomaly['type'], 1.0)
            
            # Base severity
            if anomaly.get('severity') == 'High':
                severity_multiplier = 3.0
            elif anomaly.get('severity') == 'Medium':
                severity_multiplier = 2.0
            else:
                severity_multiplier = 1.0
            
            # Value-based adjustment (for quantitative anomalies)
            value_bonus = 0.0
            if anomaly.get('value') is not None and anomaly.get('threshold') is not None:
                try:
                    ratio = abs(anomaly['value']) / anomaly['threshold']
                    if ratio > 1:
                        value_bonus = min(2.0, (ratio - 1) * 0.5)
                except:
                    pass
            
            score += (severity_multiplier * type_weight) + value_bonus
        
        # Normalize by number of anomalies
        normalized_score = score / len(anomalies)
        
        return round(normalized_score, 2)
    
    def _categorize_severity(self, score: float) -> str:
        """Categorize severity score into levels"""
        if score >= 3.0:
            return 'Critical'
        elif score >= 2.0:
            return 'High'
        elif score >= 1.0:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_lateness_severity(self, lateness_minutes: float) -> str:
        """Determine severity based on lateness minutes"""
        try:
            x = float(lateness_minutes)
        except (TypeError, ValueError):
            x = 0.0

        if x > 120:
            return 'High'
        elif x > 60:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_early_departure_severity(self, early_minutes: float) -> str:
        """Determine severity based on early departure minutes"""
        if early_minutes > 180:
            return 'High'
        elif early_minutes > 120:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_short_day_severity(self, hours: float) -> str:
        """Determine severity based on short day hours"""
        if hours < 2:
            return 'High'
        elif hours < 3:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_long_day_severity(self, hours: float) -> str:
        """Determine severity based on long day hours"""
        if hours > 14:
            return 'High'
        elif hours > 12:
            return 'Medium'
        else:
            return 'Low'
    
    def generate_anomaly_report(self, anomalies_df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive anomaly report
        
        Returns:
            Dictionary with report statistics and insights
        """
        if anomalies_df.empty:
            return {
                'total_anomalies': 0,
                'employees_affected': 0,
                'message': 'No anomalies detected'
            }
        
        report = {
            'total_anomalies': len(anomalies_df),
            'employees_affected': anomalies_df['employee_id'].nunique(),
            'date_range': {
                'start': anomalies_df['date'].min(),
                'end': anomalies_df['date'].max()
            },
            'severity_distribution': anomalies_df['severity_level'].value_counts().to_dict(),
            'department_distribution': anomalies_df['department'].value_counts().to_dict(),
            'top_anomaly_types': self._get_top_anomaly_types(anomalies_df),
            'top_offenders': self._get_top_offenders(anomalies_df),
            'daily_trend': self._get_daily_trend(anomalies_df)
        }
        
        return report
    
    def _get_top_anomaly_types(self, anomalies_df: pd.DataFrame) -> List[Dict]:
        """Get most common anomaly types"""
        all_types = []
        for types_str in anomalies_df['anomaly_types']:
            types = [t.strip() for t in types_str.split(',')]
            all_types.extend(types)
        
        from collections import Counter
        type_counts = Counter(all_types)
        
        return [
            {'type': type_name, 'count': count}
            for type_name, count in type_counts.most_common(10)
        ]
    
    def _get_top_offenders(self, anomalies_df: pd.DataFrame) -> List[Dict]:
        """Get employees with most anomalies"""
        offender_stats = []
        
        for emp_id in anomalies_df['employee_id'].dropna().unique():
            emp_data = anomalies_df[anomalies_df['employee_id'] == emp_id]
            if emp_data.empty:
                continue
            
            # Calculate scores
            total_anomalies = len(emp_data)
            avg_severity = emp_data['severity_score'].mean()
            max_severity = emp_data['severity_score'].max()
            
            # Get most common anomaly type
            all_types = []
            for types_str in emp_data['anomaly_types']:
                types = [t.strip() for t in types_str.split(',')]
                all_types.extend(types)
            
            from collections import Counter
            if all_types:
                common_type = Counter(all_types).most_common(1)[0][0]
            else:
                common_type = 'Unknown'
            
            offender_stats.append({
                'employee_id': emp_id,
                'employee_name': emp_data['employee_name'].iloc[0] if 'employee_name' in emp_data.columns else str(emp_id),
                'department': emp_data['department'].iloc[0] if 'department' in emp_data.columns else 'Unknown',
                'total_anomalies': total_anomalies,
                'avg_severity': round(avg_severity, 2),
                'max_severity': round(max_severity, 2),
                'most_common_anomaly': common_type
            })
        
        # Sort by total anomalies and severity
        offender_stats.sort(key=lambda x: (x['total_anomalies'], x['avg_severity']), reverse=True)
        
        return offender_stats[:10]  # Top 10 offenders
    
    def _get_daily_trend(self, anomalies_df: pd.DataFrame) -> List[Dict]:
        """Get daily anomaly trend"""
        daily_trend = anomalies_df.groupby('date').agg({
            'employee_id': 'count',
            'severity_score': 'mean'
        }).reset_index()
        
        daily_trend.columns = ['date', 'anomaly_count', 'avg_severity']
        daily_trend['avg_severity'] = daily_trend['avg_severity'].round(2)
        
        return daily_trend.to_dict('records')


# Helper function for quick usage
def detect_anomalies(daily_df: pd.DataFrame, config: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function for quick anomaly detection
    
    Args:
        daily_df: DataFrame from data_preparation.py
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (anomalies_df, report_dict)
    """
    detector = RuleBasedAnomalyDetector(config)
    anomalies_df = detector.detect_anomalies(daily_df)
    report = detector.generate_anomaly_report(anomalies_df)
    
    return anomalies_df, report


if __name__ == "__main__":
    # Example usage
    print("Rule-Based Anomaly Detector Module")
    print("=" * 50)
    print("Usage:")
    print("1. Import: from anomaly_detection.rule_based_detector import RuleBasedAnomalyDetector")
    print("2. Initialize: detector = RuleBasedAnomalyDetector(config)")
    print("3. Detect: anomalies = detector.detect_anomalies(daily_data)")
    print("4. Report: report = detector.generate_anomaly_report(anomalies)")
    print("\nAvailable anomaly types:")
    print("- Time-based: lateness, early departure, short/long days")
    print("- Pattern-based: check frequency, sequence issues")
    print("- Location-based: multiple locations, frequent changes")
    print("- Statistical: outliers, deviations from patterns")
    print("- Behavioral: weekend/holiday work, consecutive issues")
