# anomaly_detection/data_preparation.py
"""
Data Preparation Module for Attendance Anomaly Detection
Prepares raw attendance data for rule-based, statistical, and ML anomaly detection
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, time, timedelta
import holidays
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class AttendanceDataPreprocessor:
    """
    Phase 1: Clean and prepare attendance data for anomaly detection
    
    This class handles:
    1. Data cleaning and validation
    2. Daily summary creation
    3. Feature engineering for anomaly detection
    4. Nigerian holiday integration
    """
    
    def __init__(self, 
                 standard_start_time: str = "08:00",
                 standard_end_time: str = "17:00",
                 break_start_time: str = "13:00",
                 break_end_time: str = "14:00",
                 country: str = 'NG',
                 grace_period_minutes: int = 15,
                 overtime_threshold_hours: float = 9.0):
        """
        Initialize the preprocessor with Nigerian context.
        
        Parameters:
        -----------
        standard_start_time : str
            Expected start time (default: "08:00")
        standard_end_time : str
            Expected end time (default: "17:00")
        break_start_time : str
            Break start time (default: "13:00")
        break_end_time : str
            Break end time (default: "14:00")
        country : str
            Country code for holidays (default: 'NG' for Nigeria)
        grace_period_minutes : int
            Grace period for lateness (default: 15 minutes)
        overtime_threshold_hours : float
            Hours beyond which overtime is considered (default: 9 hours)
        """
        # Time configurations
        self.standard_start = self._parse_time(standard_start_time)
        self.standard_end = self._parse_time(standard_end_time)
        self.break_start = self._parse_time(break_start_time)
        self.break_end = self._parse_time(break_end_time)
        self.grace_period = grace_period_minutes
        self.overtime_threshold = overtime_threshold_hours
        
        # Standard work duration (8 hours including 1 hour break)
        self.standard_work_hours = 8
        
        # Initialize Nigerian holidays for 2023-2024
        self.holiday_calendar = self._initialize_holidays(country)
        
        # Cache for employee patterns
        self.employee_patterns = {}
        
    def _parse_time(self, time_str: str) -> time:
        """Convert time string to time object"""
        return datetime.strptime(time_str, "%H:%M").time()
    
    def _initialize_holidays(self, country: str) -> set:
        """Initialize holiday calendar for Nigeria"""
        try:
            ng_holidays = holidays.Nigeria(years=range(2023, 2025))
            holiday_dates = set(ng_holidays.keys())
            
            # Add common Nigerian public holidays (in case library doesn't have all)
            additional_holidays = {
                '2023-01-01', '2023-04-07', '2023-04-10', '2023-05-01',
                '2023-05-29', '2023-06-12', '2023-10-01', '2023-12-25', '2023-12-26',
                '2024-01-01', '2024-03-29', '2024-04-01', '2024-05-01',
                '2024-05-27', '2024-06-12', '2024-10-01', '2024-12-25', '2024-12-26'
            }
            
            # Convert to datetime objects
            additional_dates = {pd.to_datetime(date).date() for date in additional_holidays}
            holiday_dates.update(additional_dates)
            
            return holiday_dates
            
        except Exception as e:
            print(f"Warning: Could not load holiday calendar: {e}")
            return set()
    
    def prepare_data_pipeline(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete data preparation pipeline.
        """
        print("Starting data preparation pipeline...")
        
        # Step 1: Clean and validate raw data
        print("Step 1: Cleaning raw data...")
        cleaned_data = self.clean_and_prepare(raw_df)
        
        if 'created_at' in cleaned_data.columns:
            cleaned_data['created_at'] = pd.to_datetime(
                cleaned_data['created_at'],
                errors='coerce',
                dayfirst=True,
                format='mixed',
            )

        # DEBUG: Check cleaned data
        print(f"\nDEBUG: After cleaning - Shape: {cleaned_data.shape}")
        if not cleaned_data.empty:
            print(f"Columns: {list(cleaned_data.columns)}")
            print(f"Sample data:\n{cleaned_data.head(3).to_string()}")

        # Combine date + time into a proper timestamp
        cleaned_data['timestamp'] = pd.to_datetime(
            cleaned_data['date'].astype(str) + ' ' + cleaned_data['time'].astype(str),
            errors='coerce'
        )
        # Step 2: Create daily summaries
        print("\n=== DEBUG: Before calculate_daily_summaries ===")
        print("cleaned_data shape:", cleaned_data.shape)
        print("cleaned_data columns:", cleaned_data.columns.tolist())
        print(cleaned_data.head())

        print("\nStep 2: Creating daily summaries...")
        daily_summaries = self.calculate_daily_summaries(cleaned_data)
    
        # If daily_summaries is empty, create a fallback
        if daily_summaries is None or daily_summaries.empty:
            print("\nWARNING: daily_summaries is empty! Creating fallback structure...")
            daily_summaries = self.create_fallback_summaries(cleaned_data)
        
        # Step 3: Add anomaly features (only if we have data)
        if not daily_summaries.empty:
            print("Step 3: Adding anomaly features...")
            daily_summaries = self.add_anomaly_features(daily_summaries)
            
            # Step 4: Calculate employee patterns
            print("Step 4: Calculating employee patterns...")
            daily_summaries = self.add_employee_patterns(daily_summaries)
        else:
            print("Skipping steps 3-4: No daily summaries available")
        
        print(f"\nData preparation complete!")
        print(f"  - Cleaned records: {len(cleaned_data)}")
        print(f"  - Daily summaries: {len(daily_summaries)}")
        
        if not daily_summaries.empty:
            print(f"  - Unique employees: {daily_summaries['employee_id'].nunique()}")
        
        return cleaned_data, daily_summaries

    def create_fallback_summaries(self, cleaned_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create fallback daily summaries when normal method fails.
        """
        if cleaned_data.empty:
            return pd.DataFrame()
        
        try:
            # Ensure we have required columns
            df = cleaned_data.copy()
            
            # Add missing columns if needed
            if 'date' not in df.columns and 'created_at' in df.columns:
                df['date'] = pd.to_datetime(df['created_at']).dt.date
            
            if 'employee_id' not in df.columns:
                df['employee_id'] = df.get('employee_name', 'UNKNOWN')
            
            # Convert date to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
            
            if df.empty:
                return pd.DataFrame()
            
            # Create simple summary
            summary = df.groupby(['date', 'employee_id']).agg(
                check_count=('check_type', 'count'),
                employee_name=('employee_name', 'first')
            ).reset_index()
            
            print(f"Created fallback summaries with {len(summary)} rows")
            return summary
            
        except Exception as e:
            print(f"Error creating fallback: {e}")
            return pd.DataFrame()
    
    def clean_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
    # def clean_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Clean and prepare raw attendance data
        """
        df = df.copy()
        
        # 1. Basic data validation
        required_columns = ['employee_id', 'date', 'time', 'check_type']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 2. Handle missing values
        df = df.dropna(subset=['employee_id', 'date', 'time', 'check_type'])
        
        # 3. Combine date and time into timestamp
        try:
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
            # Drop rows where timestamp conversion failed
            df = df.dropna(subset=['timestamp'])
        except Exception as e:
            print(f"Warning: Error creating timestamp: {e}")
            # Try alternative approach
            df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['time'])
        
        # 4. Extract date for grouping (CRITICAL FIX!)
        df['date'] = df['timestamp'].dt.date  # Extract date from timestamp
        
        # 5. Sort chronologically
        df = df.sort_values(['employee_id', 'timestamp'])
        
        # 6. Remove exact duplicates
        df = df.drop_duplicates(subset=['employee_id', 'timestamp', 'check_type'])
        
        # 7. Validate check sequences
        df = self._validate_check_sequences(df)
        
        # 8. Add check number for the day
        df = self._add_check_numbers(df)
        
        print(f"\nClean and prepare completed: {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        
        return df
    
    def _validate_check_sequences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate IN/OUT sequences and flag anomalies
        """
        df['sequence_valid'] = True
        df['sequence_issue'] = ''
        
        for emp_id in df['employee_id'].unique():
            emp_mask = df['employee_id'] == emp_id
            emp_data = df[emp_mask].copy()
            
            if len(emp_data) < 2:
                continue
            
            # Check for consecutive same check types
            for i in range(1, len(emp_data)):
                idx = emp_data.index[i]
                prev_check = emp_data.iloc[i-1]['check_type']
                curr_check = emp_data.iloc[i]['check_type']
                prev_time = emp_data.iloc[i-1]['timestamp']
                curr_time = emp_data.iloc[i]['timestamp']
                
                # Check for duplicate consecutive check types
                if prev_check == curr_check:
                    df.at[idx, 'sequence_valid'] = False
                    df.at[idx, 'sequence_issue'] = f'Consecutive {curr_check}'
                
                # Check for impossible time sequences (out before in)
                if prev_check == 'check_out' and curr_check == 'check_in':
                    time_diff = (curr_time - prev_time).total_seconds() / 60  # minutes
                    if time_diff < 0:
                        df.at[idx, 'sequence_valid'] = False
                        df.at[idx, 'sequence_issue'] = 'OUT before IN'
        
        return df
    
    def _add_check_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add check sequence numbers for each employee day"""
        df['check_number'] = 0
        
        for (emp_id, date), group in df.groupby(['employee_id', df['timestamp'].dt.date]):
            mask = (df['employee_id'] == emp_id) & (df['timestamp'].dt.date == date)
            df.loc[mask, 'check_number'] = range(1, len(group) + 1)
        
        return df
    
    def calculate_daily_summaries(self, cleaned_data: pd.DataFrame) -> pd.DataFrame:
        print("\n=== DAILY SUMMARY DEBUG ===")
        print("Rows in cleaned_data:", len(cleaned_data))
        print("Columns:", cleaned_data.columns.tolist())

        if cleaned_data is None or cleaned_data.empty:
            return pd.DataFrame()

        df = cleaned_data.copy()

        # Ensure timestamp exists
        if 'timestamp' not in df.columns or df['timestamp'].isna().all():
            df['timestamp'] = pd.to_datetime(
                df['date'].astype(str) + ' ' + df['time'].astype(str),
                errors='coerce'
            )

        # Ensure date is consistent with timestamp (daily grouping)
        # Use dt.date so it's stable and matches your existing code
        df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date

        print("Timestamp nulls:", df['timestamp'].isna().sum())
        print(df[['employee_id', 'employee_name', 'timestamp', 'date']].head())

        # Normalize check_type
        df['check_type'] = df['check_type'].astype(str).str.lower().str.strip()

        # ---------------------------
        # 1) Daily base aggregation
        # ---------------------------
        # daily = (
        #     df
        #     .dropna(subset=['employee_id', 'date'])
        #     .groupby(['employee_id', 'employee_name', 'date'], as_index=False)
        #     .agg(
        #         first_check_in=('timestamp', 'min'),
        #         last_check_out=('timestamp', 'max'),
        #         total_checks=('check_type', 'count')
        #     )
        # )

        daily = (
            df
            .dropna(subset=['employee_id', 'date'])
            .groupby(['employee_id', 'employee_name', 'date'], as_index=False)
            .agg(
                department=('department', 'first'),
                job_title=('job_title', 'first'),
                first_check_in=('timestamp', 'min'),
                last_check_out=('timestamp', 'max'),
                total_checks=('check_type', 'count')
            )
        )

        daily['total_checks'] = pd.to_numeric(daily['total_checks'], errors='coerce').fillna(0).astype(int)

        # Work duration in hours
        daily['work_duration_hours'] = (
            (pd.to_datetime(daily['last_check_out']) - pd.to_datetime(daily['first_check_in']))
            .dt.total_seconds() / 3600
        )
        daily['work_duration_hours'] = daily['work_duration_hours'].fillna(0).clip(lower=0)

        # ---------------------------
        # 2) Incomplete pairs
        # ---------------------------
        counts = (
            df
            .pivot_table(
                index=['employee_id', 'employee_name', 'date'],
                columns='check_type',
                values='timestamp',
                aggfunc='count',
                fill_value=0
            )
            .reset_index()
        )

        # Ensure columns exist
        if 'check_in' not in counts.columns:
            counts['check_in'] = 0
        if 'check_out' not in counts.columns:
            counts['check_out'] = 0

        counts['has_incomplete_pairs'] = counts['check_in'] != counts['check_out']

        daily = daily.merge(
            counts[['employee_id', 'employee_name', 'date', 'has_incomplete_pairs']],
            on=['employee_id', 'employee_name', 'date'],
            how='left'
        )
        daily['has_incomplete_pairs'] = daily['has_incomplete_pairs'].fillna(False).astype(bool)

        # ---------------------------
        # 3) Sequence issues count
        # ---------------------------
        if 'sequence_valid' in df.columns:
            df['sequence_issue_flag'] = (~df['sequence_valid'].astype(bool)).astype(int)
        elif 'sequence_issue' in df.columns:
            df['sequence_issue_flag'] = df['sequence_issue'].astype(str).str.strip().ne("").astype(int)
        else:
            df['sequence_issue_flag'] = 0

        seq = (
            df
            .groupby(['employee_id', 'employee_name', 'date'], as_index=False)
            .agg(sequence_issues_count=('sequence_issue_flag', 'sum'))
        )

        daily = daily.merge(
            seq[['employee_id', 'employee_name', 'date', 'sequence_issues_count']],
            on=['employee_id', 'employee_name', 'date'],
            how='left'
        )
        daily['sequence_issues_count'] = daily['sequence_issues_count'].fillna(0).astype(int)

        # ---------------------------
        # 4) Location features
        # ---------------------------
        def _safe_str(x):
            return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x).strip()

        if {'location_city', 'location_region', 'location_country'}.issubset(df.columns):
            df['location_label'] = df.apply(
                lambda r: ", ".join([p for p in [
                    _safe_str(r.get('location_city')),
                    _safe_str(r.get('location_region')),
                    _safe_str(r.get('location_country'))
                ] if p]) or "Unknown",
                axis=1
            )
        else:
            df['location_label'] = "Unknown"

        loc_daily = (
            df
            .sort_values('timestamp')
            .groupby(['employee_id', 'employee_name', 'date'])['location_label']
            .apply(list)
            .reset_index(name='location_seq')
        )

        loc_daily['unique_locations'] = loc_daily['location_seq'].apply(lambda xs: len(set(xs)))
        loc_daily['has_multiple_locations'] = loc_daily['unique_locations'] > 1
        loc_daily['location_changes'] = loc_daily['location_seq'].apply(
            lambda xs: sum(1 for i in range(1, len(xs)) if xs[i] != xs[i - 1])
        )

        daily = daily.merge(
            loc_daily[['employee_id', 'employee_name', 'date',
                    'has_multiple_locations', 'location_changes', 'unique_locations']],
            on=['employee_id', 'employee_name', 'date'],
            how='left'
        )

        daily['has_multiple_locations'] = daily['has_multiple_locations'].fillna(False).astype(bool)
        daily['location_changes'] = daily['location_changes'].fillna(0).astype(int)
        daily['unique_locations'] = daily['unique_locations'].fillna(0).astype(int)

        print("\n=== DAILY SUMMARY RESULT (preview) ===")
        print(daily.head())
        print("Columns:", daily.columns.tolist())

        return daily
    
    def _process_daily_attendance(self, daily_checks: pd.DataFrame, 
                             emp_id: str, date_val: date) -> Dict:
        """
        Process a single day's attendance for one employee
        """
        # Sort by timestamp
        daily_checks = daily_checks.sort_values('timestamp')
        
        # Separate check types
        check_ins = daily_checks[daily_checks['check_type'] == 'check_in']
        check_outs = daily_checks[daily_checks['check_type'] == 'check_out']
        
        # Basic metrics
        first_check_in = check_ins['timestamp'].min() if not check_ins.empty else None
        last_check_out = check_outs['timestamp'].max() if not check_outs.empty else None
        
        # Calculate work duration (handle multiple IN/OUT pairs)
        work_duration = self._calculate_daily_work_duration(check_ins, check_outs)
        
        # Location analysis
        location_metrics = self._analyze_locations(daily_checks)
        
        # Sequence issues
        sequence_issues = daily_checks[~daily_checks['sequence_valid']]
        
        # Build result dictionary
        result = {
            'employee_id': emp_id,
            'employee_name': daily_checks['employee_name'].iloc[0] if 'employee_name' in daily_checks.columns else emp_id,
            'department': daily_checks['department'].iloc[0] if 'department' in daily_checks.columns else 'Unknown',
            'job_title': daily_checks['job_title'].iloc[0] if 'job_title' in daily_checks.columns else 'Unknown',
            'date': date_val,
            'first_check_in': first_check_in,
            'last_check_out': last_check_out,
            'total_checks': len(daily_checks),
            'check_ins_count': len(check_ins),
            'check_outs_count': len(check_outs),
            'work_duration_hours': work_duration,
            'has_incomplete_pairs': len(check_ins) != len(check_outs),
            'sequence_issues_count': len(sequence_issues),
            'sequence_issues': '; '.join(sequence_issues['sequence_issue'].unique()) if not sequence_issues.empty else '',
        }
        
        # Add location metrics
        result.update(location_metrics)
        
        return result
    
    def _calculate_daily_work_duration(self, check_ins: pd.DataFrame, 
                                      check_outs: pd.DataFrame) -> float:
        """
        Calculate total work duration for a day (in hours)
        Handles multiple IN/OUT pairs
        """
        if len(check_ins) == 0 or len(check_outs) == 0:
            return 0.0
        
        # Align check-ins and check-outs
        total_minutes = 0
        
        # Pair check-ins with corresponding check-outs
        min_pairs = min(len(check_ins), len(check_outs))
        
        for i in range(min_pairs):
            in_time = check_ins.iloc[i]['timestamp']
            out_time = check_outs.iloc[i]['timestamp']
            
            # Ensure out_time is after in_time
            if out_time > in_time:
                duration = (out_time - in_time).total_seconds() / 3600  # hours
                total_minutes += duration
        
        return round(total_minutes, 2)
    
    def _analyze_locations(self, daily_checks: pd.DataFrame) -> Dict:
        """Analyze location patterns for a day"""
        if 'location_city' not in daily_checks.columns:
            return {
                'unique_locations': 0,
                'location_changes': 0,
                'main_location': 'Unknown',
                'has_multiple_locations': False
            }
        
        # Get unique locations
        locations = daily_checks['location_city'].dropna().unique()
        unique_locations = len(locations)
        
        # Count location changes
        location_changes = 0
        location_sequence = daily_checks['location_city'].dropna().tolist()
        
        for i in range(1, len(location_sequence)):
            if location_sequence[i] != location_sequence[i-1]:
                location_changes += 1
        
        # Determine main location (most frequent)
        if len(location_sequence) > 0:
            main_location = max(set(location_sequence), key=location_sequence.count)
        else:
            main_location = 'Unknown'
        
        return {
            'unique_locations': unique_locations,
            'location_changes': location_changes,
            'main_location': main_location,
            'has_multiple_locations': unique_locations > 1
        }
    
    def add_anomaly_features(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Add features for anomaly detection
        """
        if daily_df.empty:
            return daily_df
        
        daily_df = daily_df.copy()
        
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(daily_df['date']):
            daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        # 1. TEMPORAL FEATURES
        daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
        daily_df['day_name'] = daily_df['date'].dt.day_name()
        daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6])  # Saturday, Sunday
        daily_df['is_weekday'] = ~daily_df['is_weekend']
        
        # Nigerian holidays
        daily_df['is_holiday'] = daily_df['date'].apply(
            lambda x: x.date() in self.holiday_calendar
        )
        
        # 2. TIME-BASED FEATURES
        # Arrival time analysis
        daily_df['arrival_time'] = daily_df['first_check_in'].apply(
            lambda x: x.time() if pd.notna(x) else None
        )
        daily_df['arrival_hour'] = daily_df['first_check_in'].dt.hour
        daily_df['arrival_minute'] = daily_df['first_check_in'].dt.minute
        daily_df['arrival_total_minutes'] = daily_df.apply(
            lambda row: row['arrival_hour'] * 60 + row['arrival_minute'] 
            if pd.notna(row['arrival_hour']) else None, axis=1
        )
        
        # Lateness calculation (with grace period)
        daily_df['lateness_minutes'] = daily_df.apply(
            lambda row: self._calculate_lateness_with_grace(row['first_check_in']), axis=1
        )
        daily_df['is_late'] = daily_df['lateness_minutes'] > 0
        
        # Departure time analysis
        daily_df['departure_time'] = daily_df['last_check_out'].apply(
            lambda x: x.time() if pd.notna(x) else None
        )
        daily_df['departure_hour'] = daily_df['last_check_out'].dt.hour
        daily_df['departure_minute'] = daily_df['last_check_out'].dt.minute
        daily_df['departure_total_minutes'] = daily_df.apply(
            lambda row: row['departure_hour'] * 60 + row['departure_minute']
            if pd.notna(row['departure_hour']) else None, axis=1
        )
        
        # Early departure calculation
        daily_df['early_departure_minutes'] = daily_df.apply(
            lambda row: self._calculate_early_departure(row['last_check_out']), axis=1
        )
        daily_df['is_early_departure'] = daily_df['early_departure_minutes'] > 0
        
        # 3. DURATION-BASED FEATURES
        # Guard against extreme outliers that can dominate model features
        daily_df['work_duration_hours'] = daily_df['work_duration_hours'].clip(lower=0, upper=16)

        daily_df['short_day_flag'] = daily_df['work_duration_hours'] < (self.standard_work_hours - 2)
        daily_df['long_day_flag'] = daily_df['work_duration_hours'] > (self.standard_work_hours + 3)
        daily_df['very_long_day_flag'] = daily_df['work_duration_hours'] > 12
        daily_df['overtime_hours'] = daily_df['work_duration_hours'].apply(
            lambda x: max(0, x - self.overtime_threshold)
        )
        daily_df['has_overtime'] = daily_df['overtime_hours'] > 0

        # Low-effort ratio features that improve separation
        daily_df['checks_per_hour'] = daily_df['total_checks'] / (daily_df['work_duration_hours'] + 0.1)
        daily_df['lateness_ratio'] = daily_df['lateness_minutes'] / (daily_df['work_duration_hours'] + 0.1)

        # Coarse time bins for reduced noise
        daily_df['arrival_time_bin'] = pd.cut(
            daily_df['arrival_hour'],
            bins=[-1, 6, 8, 10, 23],
            labels=['very_early', 'early', 'on_time', 'late']
        ).astype(str)
        daily_df['departure_time_bin'] = pd.cut(
            daily_df['departure_hour'],
            bins=[-1, 14, 16, 18, 23],
            labels=['very_early', 'early', 'on_time', 'late']
        ).astype(str)

        # Department-level normalization (quick group context)
        if 'department' in daily_df.columns:
            dept_hours_mean = daily_df.groupby('department')['work_duration_hours'].transform('mean')
            dept_hours_std = daily_df.groupby('department')['work_duration_hours'].transform('std').replace(0, np.nan)
            daily_df['dept_hours_z'] = (daily_df['work_duration_hours'] - dept_hours_mean) / dept_hours_std

            dept_late_mean = daily_df.groupby('department')['lateness_minutes'].transform('mean')
            dept_late_std = daily_df.groupby('department')['lateness_minutes'].transform('std').replace(0, np.nan)
            daily_df['dept_lateness_z'] = (daily_df['lateness_minutes'] - dept_late_mean) / dept_late_std
        
        # 4. PATTERN-BASED FEATURES
        daily_df['check_frequency_flag'] = daily_df['total_checks'].apply(
            lambda x: 'High' if x > 6 else ('Low' if x < 2 else 'Normal')
        )
        daily_df['missing_pair_flag'] = daily_df['has_incomplete_pairs']
        daily_df['sequence_issue_flag'] = daily_df['sequence_issues_count'] > 0
        
        # 5. LOCATION-BASED FEATURES
        daily_df['location_anomaly_flag'] = daily_df['has_multiple_locations']
        daily_df['frequent_location_changes'] = daily_df['location_changes'] > 2
        
        # 6. COMPOSITE FLAGS
        daily_df['has_time_anomaly'] = (
            daily_df['is_late'] | 
            daily_df['is_early_departure'] |
            daily_df['short_day_flag'] |
            daily_df['long_day_flag']
        )
        
        daily_df['has_pattern_anomaly'] = (
            daily_df['missing_pair_flag'] |
            daily_df['sequence_issue_flag'] |
            (daily_df['check_frequency_flag'] != 'Normal')
        )
        
        daily_df['has_location_anomaly'] = (
            daily_df['location_anomaly_flag'] |
            daily_df['frequent_location_changes']
        )
        
        daily_df['total_anomaly_flags'] = (
            daily_df['has_time_anomaly'].astype(int) +
            daily_df['has_pattern_anomaly'].astype(int) +
            daily_df['has_location_anomaly'].astype(int)
        )
        
        daily_df['anomaly_severity'] = daily_df['total_anomaly_flags'].apply(
            lambda x: 'High' if x >= 2 else ('Medium' if x == 1 else 'Low')
        )
        
        return daily_df
    
    def _calculate_lateness_with_grace(self, check_in_time: pd.Timestamp) -> float:
        """Calculate lateness in minutes with grace period"""
        if pd.isna(check_in_time):
            return 0.0
        
        check_in_dt = check_in_time.to_pydatetime()
        standard_start_dt = datetime.combine(check_in_dt.date(), self.standard_start)
        
        # Add grace period
        grace_end = standard_start_dt + timedelta(minutes=self.grace_period)
        
        # Calculate minutes late (after grace period)
        if check_in_dt > grace_end:
            lateness = (check_in_dt - grace_end).total_seconds() / 60
            return round(lateness, 1)
        
        return 0.0
    
    def _calculate_early_departure(self, check_out_time: pd.Timestamp) -> float:
        """Calculate early departure in minutes"""
        if pd.isna(check_out_time):
            return 0.0
        
        check_out_dt = check_out_time.to_pydatetime()
        standard_end_dt = datetime.combine(check_out_dt.date(), self.standard_end)
        
        # Calculate minutes early (before standard end)
        if check_out_dt < standard_end_dt:
            early_departure = (standard_end_dt - check_out_dt).total_seconds() / 60
            return round(early_departure, 1)
        
        return 0.0
    
    def add_employee_patterns(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Add employee-specific patterns for comparison
        Calculates rolling averages and personal baselines
        """
        if daily_df.empty:
            return daily_df
        
        daily_df = daily_df.copy()

        # Make sure required numeric columns exist and are numeric
        for col in ['arrival_total_minutes', 'departure_total_minutes', 'work_duration_hours']:
            if col not in daily_df.columns:
                daily_df[col] = np.nan
            daily_df[col] = pd.to_numeric(daily_df[col], errors='coerce')
        
        # Sort for rolling calculations
        daily_df = daily_df.sort_values(['employee_id', 'date'])
        
        # Initialize lists for results
        arrival_rolling_avg = []
        departure_rolling_avg = []
        hours_rolling_avg = []
        arrival_z_score = []
        hours_z_score = []
        
        # Calculate for each employee
        for emp_id in daily_df['employee_id'].unique():
            emp_mask = daily_df['employee_id'] == emp_id
            emp_data = daily_df[emp_mask].copy()
            
            if len(emp_data) < 3:
                # Not enough data for rolling averages
                arrival_rolling_avg.extend([0.0] * len(emp_data))
                departure_rolling_avg.extend([0.0] * len(emp_data))
                hours_rolling_avg.extend([0.0] * len(emp_data))
                arrival_z_score.extend([0.0] * len(emp_data))
                hours_z_score.extend([0.0] * len(emp_data))
                continue
            
            # Calculate rolling averages (7-day window)
            arrival_minutes = emp_data['arrival_total_minutes'].rolling(window=7, min_periods=3).mean()
            departure_minutes = emp_data['departure_total_minutes'].rolling(window=7, min_periods=3).mean()
            work_hours = emp_data['work_duration_hours'].rolling(window=7, min_periods=3).mean()
            
            arrival_rolling_avg.extend(arrival_minutes.tolist())
            departure_rolling_avg.extend(departure_minutes.tolist())
            hours_rolling_avg.extend(work_hours.tolist())
            
            # Calculate Z-scores for statistical anomalies
            arrival_mean = emp_data['arrival_total_minutes'].mean()
            arrival_std = emp_data['arrival_total_minutes'].std()
            hours_mean = emp_data['work_duration_hours'].mean()
            hours_std = emp_data['work_duration_hours'].std()
            
            #if arrival_std > 0:
            if pd.notna(arrival_std) and arrival_std > 0:
                arr_z = (emp_data['arrival_total_minutes'] - arrival_mean) / arrival_std
                arrival_z_score.extend(arr_z.tolist())
            else:
                arrival_z_score.extend([0] * len(emp_data))
            
            #if hours_std > 0:
            if pd.notna(hours_std) and hours_std > 0:
                hr_z = (emp_data['work_duration_hours'] - hours_mean) / hours_std
                hours_z_score.extend(hr_z.tolist())
            else:
                hours_z_score.extend([0] * len(emp_data))
        
        # Add calculated features
        daily_df['arrival_rolling_avg_7d'] = arrival_rolling_avg
        daily_df['departure_rolling_avg_7d'] = departure_rolling_avg
        daily_df['hours_rolling_avg_7d'] = hours_rolling_avg
        daily_df['arrival_z_score'] = arrival_z_score
        daily_df['hours_z_score'] = hours_z_score

        # Flag statistical anomalies (|Z| > 2)
        daily_df['arrival_statistical_anomaly'] = daily_df['arrival_z_score'].abs() > 2
        daily_df['hours_statistical_anomaly'] = daily_df['hours_z_score'].abs() > 2
        
        return daily_df


# Helper function for quick usage
def prepare_attendance_data(raw_df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for quick data preparation
    
    Args:
        raw_df: Raw attendance DataFrame
        **kwargs: Additional arguments for AttendanceDataPreprocessor
        
    Returns:
        Tuple of (cleaned_data, daily_summaries)
    """
    preprocessor = AttendanceDataPreprocessor(**kwargs)
    return preprocessor.prepare_data_pipeline(raw_df)


if __name__ == "__main__":
    # Example usage
    print("Attendance Data Preprocessor Module")
    print("Usage:")
    print("1. Import: from anomaly_detection.data_preparation import AttendanceDataPreprocessor")
    print("2. Initialize: preprocessor = AttendanceDataPreprocessor()")
    print("3. Prepare: cleaned, daily = preprocessor.prepare_data_pipeline(your_dataframe)")
