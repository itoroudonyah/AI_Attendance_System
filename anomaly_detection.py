# anomaly_detection.py (COMPLETE CODE - Syntax Fix for String Literals)
import pandas as pd
from datetime import datetime, timedelta, time
import numpy as np
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler # (Scaler is not used in this version but often accompanies ML models)

# Ensure database functions are correctly imported/defined
import database as db

# Define model path
MODEL_PATH = 'ml_models/anomaly_model.joblib'
# Ensure ml_models directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def train_anomaly_model(re_train=False, attendance_df=None, employees_df=None):
    """
    Trains or retrains an ML anomaly detection model.
    """
    try:
        if attendance_df is None:
            attendance_df = db.get_all_attendance()
        if employees_df is None:
            employees_df = db.get_all_employees()

        if attendance_df.empty:
            return False, "Not enough attendance data to train the model."
        if employees_df.empty:
            return False, "No employee data found."

        if 'timestamp' not in attendance_df.columns:
            return False, "Attendance data missing 'timestamp' column."
        
        attendance_df['timestamp'] = pd.to_datetime(attendance_df['timestamp'])

        # Feature engineering
        attendance_df['hour'] = attendance_df['timestamp'].dt.hour
        attendance_df['minute'] = attendance_df['timestamp'].dt.minute
        attendance_df['day_of_week'] = attendance_df['timestamp'].dt.dayofweek
        attendance_df['day_of_year'] = attendance_df['timestamp'].dt.dayofyear

        features = ['hour', 'minute', 'day_of_week', 'day_of_year']
        X = attendance_df[features]
        X = X.dropna()
        
        if X.empty:
            return False, "No valid features extracted from attendance data for training."

        # CRITICAL: Use a very low contamination rate
        # Start with 1% expected anomalies, not 'auto'
        model = IsolationForest(
            random_state=42, 
            contamination=0.01,  # Only 1% expected anomalies
            n_estimators=100,
            max_samples=0.8,
            bootstrap=False,
            verbose=1
        )
        model.fit(X)

        # Test the model on training data to see what % it flags
        predictions = model.predict(X)
        anomaly_count = (predictions == -1).sum()
        anomaly_percentage = (anomaly_count / len(predictions)) * 100
        
        print(f"[DEBUG AD train_anomaly_model] Model flags {anomaly_count}/{len(predictions)} ({anomaly_percentage:.2f}%) as anomalies on training data")

        joblib.dump(model, MODEL_PATH)
        print(f"[DEBUG AD train_anomaly_model] ML model trained and saved to {MODEL_PATH}.")
        return True, f"ML Anomaly Model trained successfully! Flags {anomaly_percentage:.1f}% as anomalies."

    except Exception as e:
        print(f"[ERROR AD train_anomaly_model] Error training ML model: {e}")
        return False, f"Error training ML model: {e}"

def detect_missing_attendance(target_date, attendance_df=None, employees_df=None):
    """
    Detects employees who did not register any attendance for the target_date.
    Returns a DataFrame with details of missing employees, including department and designation.
    """
    # Fetch data if not provided
    if attendance_df is None:
        attendance_df = db.get_all_attendance()
    if employees_df is None:
        employees_df = db.get_all_employees()

    if employees_df.empty:
        print("[DEBUG AD detect_missing] No employees found in database to check for missing attendance.")
        return pd.DataFrame()

    # Find timestamp column
    timestamp_col = None
    for col in ['timestamp', 'Timestamp', 'timestamp_dt']:
        if col in attendance_df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print("[ERROR AD detect_missing] 'timestamp' column not found in attendance data. Cannot detect missing attendance.")
        return pd.DataFrame()
    
    try:
        attendance_df['timestamp_dt'] = pd.to_datetime(attendance_df[timestamp_col])
        attendance_df['attendance_date'] = attendance_df['timestamp_dt'].dt.date
    except Exception as e:
        print(f"[ERROR AD detect_missing] Error converting attendance timestamp: {e}")
        return pd.DataFrame()

    # Filter attendance records for the target date
    attendance_on_target_date = attendance_df[attendance_df['attendance_date'] == target_date]

    # Get IDs of employees who recorded attendance on the target date
    employee_id_col = 'employee_id'
    if 'employee_id' not in attendance_on_target_date.columns:
        # Try to find employee ID column
        for col in ['employee_id', 'Employee ID', 'employee_Id']:
            if col in attendance_on_target_date.columns:
                attendance_on_target_date['employee_id'] = attendance_on_target_date[col]
                employee_id_col = 'employee_id'
                break
    
    if employee_id_col not in attendance_on_target_date.columns:
        print("[ERROR AD detect_missing] Could not find employee ID column in attendance data.")
        return pd.DataFrame()
    
    present_employee_ids = attendance_on_target_date[employee_id_col].unique()

    # Identify employees who are in the employee master list but not in attendance for the target date
    missing_employees_df = employees_df[~employees_df['employee_id'].isin(present_employee_ids)].copy()

    if not missing_employees_df.empty:
        missing_employees_df['missing_dates'] = str(target_date)
        
        # Define the columns that should be in the final output DataFrame
        final_cols = [
            'employee_id',
            'full_name',
            'job_title',
            'department',
            'missing_dates'
        ]
        
        # Filter to only include columns that are actually present in the DataFrame
        existing_final_cols = [col for col in final_cols if col in missing_employees_df.columns]
        
        # Handle alternative column names
        if 'full_name' not in existing_final_cols and 'employee_name' in missing_employees_df.columns:
            missing_employees_df['full_name'] = missing_employees_df['employee_name']
            existing_final_cols.append('full_name')
        
        if 'job_title' not in existing_final_cols and 'designation' in missing_employees_df.columns:
            missing_employees_df['job_title'] = missing_employees_df['designation']
            existing_final_cols.append('job_title')
        
        if 'department' not in existing_final_cols and 'employee_department' in missing_employees_df.columns:
            missing_employees_df['department'] = missing_employees_df['employee_department']
            existing_final_cols.append('department')

        return missing_employees_df[existing_final_cols]
    else:
        print(f"[DEBUG AD detect_missing] No missing employees detected for {target_date}.")
        return pd.DataFrame()


def detect_outlier_check_ins(target_date, expected_start_time, expected_end_time, buffer_minutes, attendance_df=None, employees_df=None):
    """
    Detects attendance records that fall outside a specified time range (rule-based).
    Returns a DataFrame with details of outlier check-ins including employee info.
    """
    # Fetch data if not provided
    if attendance_df is None:
        attendance_df = db.get_all_attendance()
    if employees_df is None:
        employees_df = db.get_all_employees()

    if attendance_df.empty:
        print("[DEBUG AD detect_outlier] No attendance data for outlier check-ins.")
        return pd.DataFrame()
    if employees_df.empty:
        print("[DEBUG AD detect_outlier] No employee data for outlier check-ins (cannot merge employee details).")
        return pd.DataFrame()

    # Ensure timestamp column exists
    timestamp_col = None
    for col in ['timestamp', 'Timestamp', 'timestamp_dt']:
        if col in attendance_df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print("[ERROR AD detect_outlier] No timestamp column found in attendance data for outlier check-ins.")
        return pd.DataFrame()

    try:
        # Convert timestamp to datetime
        attendance_df['timestamp_dt'] = pd.to_datetime(attendance_df[timestamp_col])
        attendance_on_target_date = attendance_df[attendance_df['timestamp_dt'].dt.date == target_date].copy()
    except Exception as e:
        print(f"[ERROR AD detect_outlier] Error converting attendance timestamp for outlier check-ins: {e}")
        return pd.DataFrame()

    if attendance_on_target_date.empty:
        print(f"[DEBUG AD detect_outlier] No attendance records for outlier check-ins on {target_date}.")
        return pd.DataFrame()

    # Calculate buffer times
    start_buffer_dt = (datetime.combine(target_date, expected_start_time) - timedelta(minutes=buffer_minutes)).time()
    end_buffer_dt = (datetime.combine(target_date, expected_end_time) + timedelta(minutes=buffer_minutes)).time()

    # Identify outliers
    outlier_mask = (attendance_on_target_date['timestamp_dt'].dt.time < start_buffer_dt) | \
                   (attendance_on_target_date['timestamp_dt'].dt.time > end_buffer_dt)

    outlier_check_ins_df = attendance_on_target_date[outlier_mask].copy()

    if outlier_check_ins_df.empty:
        print(f"[DEBUG AD detect_outlier] No rule-based outlier check-ins detected for {target_date}.")
        return pd.DataFrame()

    # Merge with employee details
    employee_cols_to_merge = ['employee_id', 'full_name', 'job_title', 'department']
    actual_employee_cols = [col for col in employee_cols_to_merge if col in employees_df.columns]

    merged_outliers_df = pd.merge(
        outlier_check_ins_df,
        employees_df[actual_employee_cols],
        on='employee_id',
        how='left'
    )
    
    # Add a helpful column
    merged_outliers_df['Anomaly Type'] = 'Rule-Based Outlier'

    # Select and rename/reorder final columns for consistency
    final_outlier_cols = [
        'employee_id', 'full_name', 'timestamp_dt', 
        'job_title', 'department', 'Anomaly Type'
    ]
    
    # Ensure all required columns exist
    existing_final_outlier_cols = []
    for col in final_outlier_cols:
        if col in merged_outliers_df.columns:
            existing_final_outlier_cols.append(col)
        elif col == 'full_name' and 'employee_name' in merged_outliers_df.columns:
            merged_outliers_df['full_name'] = merged_outliers_df['employee_name']
            existing_final_outlier_cols.append('full_name')
        elif col == 'job_title' and 'designation' in merged_outliers_df.columns:
            merged_outliers_df['job_title'] = merged_outliers_df['designation']
            existing_final_outlier_cols.append('job_title')
    
    # Rename columns for consistency
    if 'timestamp_dt' in merged_outliers_df.columns:
        merged_outliers_df = merged_outliers_df.rename(columns={'timestamp_dt': 'Timestamp'})
    
    # Create final display columns
    display_cols = ['employee_id', 'Timestamp', 'Anomaly Type']
    if 'full_name' in merged_outliers_df.columns:
        display_cols.append('full_name')
    if 'job_title' in merged_outliers_df.columns:
        display_cols.append('job_title')
    if 'department' in merged_outliers_df.columns:
        display_cols.append('department')
    
    # Only include columns that actually exist
    final_display_cols = [col for col in display_cols if col in merged_outliers_df.columns]
    
    print(f"[DEBUG AD detect_outlier] Rule-based outliers found: {len(merged_outliers_df)} records")
    if not merged_outliers_df.empty:
        print(f"[DEBUG AD detect_outlier] Available columns: {merged_outliers_df.columns.tolist()}")
    
    return merged_outliers_df[final_display_cols]

    """
    Predicts anomalies using the trained ML model for the target_date.
    Returns a DataFrame of detected anomalies including employee details.
    """
    # 1. Load the trained model
    if not os.path.exists(MODEL_PATH):
        print(f"[DEBUG AD predict_ml] ML model not found at {MODEL_PATH}. Cannot predict anomalies.")
        return pd.DataFrame()
    
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[DEBUG AD predict_ml] ML model loaded successfully from {MODEL_PATH}.")
    except Exception as e:
        print(f"[ERROR AD predict_ml] Failed to load ML model: {e}")
        return pd.DataFrame()

    # 2. Get all attendance and employee data
    if attendance_df is None:
        attendance_df = db.get_all_attendance()
    if employees_df is None:
        employees_df = db.get_all_employees()

    if attendance_df.empty:
        print("[DEBUG AD predict_ml] No attendance data available for ML prediction.")
        return pd.DataFrame()
    if employees_df.empty:
        print("[DEBUG AD predict_ml] No employee data available for ML prediction (cannot merge employee details).")
        return pd.DataFrame()

    # Find timestamp column
    timestamp_col = None
    for col in ['timestamp', 'Timestamp', 'timestamp_dt']:
        if col in attendance_df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print("[ERROR AD predict_ml] 'timestamp' column not found in attendance data for ML prediction.")
        return pd.DataFrame()

    try:
        attendance_df['timestamp_dt'] = pd.to_datetime(attendance_df[timestamp_col])
        attendance_for_target_date = attendance_df[attendance_df['timestamp_dt'].dt.date == target_date].copy()
    except Exception as e:
        print(f"[ERROR AD predict_ml] Error converting attendance timestamp for ML prediction: {e}")
        return pd.DataFrame()

    print(f"[DEBUG AD predict_ml] Raw attendance records for {target_date}: {len(attendance_for_target_date)} rows.")
    if attendance_for_target_date.empty:
        print(f"[DEBUG AD predict_ml] No attendance records for ML prediction on {target_date}.")
        return pd.DataFrame()

    # 3. Feature Engineering (must match features used during training)
    attendance_for_target_date['hour'] = attendance_for_target_date['timestamp_dt'].dt.hour
    attendance_for_target_date['minute'] = attendance_for_target_date['timestamp_dt'].dt.minute
    attendance_for_target_date['day_of_week'] = attendance_for_target_date['timestamp_dt'].dt.dayofweek
    attendance_for_target_date['day_of_year'] = attendance_for_target_date['timestamp_dt'].dt.dayofyear
    
    # The actual features used by IsolationForest must be numeric
    model_features = ['hour', 'minute', 'day_of_week', 'day_of_year']
    
    # Ensure these features exist and are numeric before proceeding
    for feature in model_features:
        if feature not in attendance_for_target_date.columns:
            print(f"[ERROR AD predict_ml] Missing feature '{feature}' for ML prediction.")
            return pd.DataFrame()

    X_predict = attendance_for_target_date[model_features]

    # 4. Predict anomaly scores
    try:
        attendance_for_target_date['Anomaly Score'] = model.decision_function(X_predict)
        attendance_for_target_date['Is Anomaly'] = model.predict(X_predict)
        print(f"[DEBUG AD predict_ml] Anomaly scores predicted. Sample scores (Anomaly Score, Is Anomaly):\n"
              f"{attendance_for_target_date[['Anomaly Score', 'Is Anomaly']].head().to_string()}")
    except Exception as e:
        print(f"[ERROR AD predict_ml] Error predicting anomalies with ML model: {e}")
        return pd.DataFrame()

    # Filter for anomalies (-1 indicates anomaly by Isolation Forest)
    anomalies_df = attendance_for_target_date[attendance_for_target_date['Is Anomaly'] == -1].copy()

    print(f"[DEBUG AD predict_ml] Number of raw anomalies detected (Is Anomaly == -1): {len(anomalies_df)}")

    if anomalies_df.empty:
        print(f"[DEBUG AD predict_ml] No raw ML anomalies detected for {target_date}.")
        return pd.DataFrame()

    # 5. Merge with employee details to get department and designation
    employee_cols_to_merge = ['employee_id', 'full_name', 'job_title', 'department']
    actual_employee_cols = [col for col in employee_cols_to_merge if col in employees_df.columns]

    merged_anomalies_df = pd.merge(
        anomalies_df,
        employees_df[actual_employee_cols],
        on='employee_id',
        how='left'
    )

    # Handle duplicate column names created by pandas
    if 'full_name_y' in merged_anomalies_df.columns:
        merged_anomalies_df['full_name'] = merged_anomalies_df['full_name_y']
        merged_anomalies_df = merged_anomalies_df.drop(columns=['full_name_x', 'full_name_y'], errors='ignore')
    elif 'full_name_x' in merged_anomalies_df.columns:
        merged_anomalies_df['full_name'] = merged_anomalies_df['full_name_x']
        merged_anomalies_df = merged_anomalies_df.drop(columns=['full_name_x'], errors='ignore')
    
    # Rename timestamp column for consistency
    if 'timestamp_dt' in merged_anomalies_df.columns:
        merged_anomalies_df = merged_anomalies_df.rename(columns={'timestamp_dt': 'Timestamp'})

    # Select final columns for output
    final_ml_anomaly_cols = [
        'employee_id', 'full_name', 'Timestamp',
        'Anomaly Score', 'Is Anomaly',
        'job_title', 'department'
    ]
    
    # Only include columns that exist
    existing_final_ml_anomaly_cols = [col for col in final_ml_anomaly_cols if col in merged_anomalies_df.columns]

    return merged_anomalies_df[existing_final_ml_anomaly_cols]

def predict_ml_anomalies(target_date, attendance_df=None, employees_df=None):
    """
    Predicts anomalies using the trained ML model with employee context.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"[DEBUG AD predict_ml] ML model not found.")
        return pd.DataFrame()
    
    try:
        loaded_data = joblib.load(MODEL_PATH)
        model = loaded_data['model']
        scaler = loaded_data['scaler']
        dept_mapping = loaded_data['dept_mapping']
        title_mapping = loaded_data['title_mapping']
        features = loaded_data['features']
        print(f"[DEBUG AD predict_ml] ML model loaded with {len(features)} features.")
    except Exception as e:
        print(f"[ERROR AD predict_ml] Failed to load ML model: {e}")
        return pd.DataFrame()

    # Get data
    if attendance_df is None:
        attendance_df = db.get_all_attendance()
    if employees_df is None:
        employees_df = db.get_all_employees()

    # Find timestamp column
    timestamp_col = None
    for col in ['timestamp', 'Timestamp', 'timestamp_dt']:
        if col in attendance_df.columns:
            timestamp_col = col
            break
    
    if timestamp_col is None:
        print("[ERROR AD predict_ml] No timestamp column found.")
        return pd.DataFrame()

    try:
        attendance_df['timestamp_dt'] = pd.to_datetime(attendance_df[timestamp_col])
        attendance_for_target_date = attendance_df[attendance_df['timestamp_dt'].dt.date == target_date].copy()
    except Exception as e:
        print(f"[ERROR AD predict_ml] Error converting timestamp: {e}")
        return pd.DataFrame()

    if attendance_for_target_date.empty:
        print(f"[DEBUG AD predict_ml] No attendance records for {target_date}.")
        return pd.DataFrame()

    # --- ENRICH WITH EMPLOYEE DATA ---
    enriched_data = pd.merge(
        attendance_for_target_date,
        employees_df[['employee_id', 'job_title', 'department', 'hire_date']],
        on='employee_id',
        how='left'
    )
    
    # Fill missing values
    enriched_data['job_title'] = enriched_data['job_title'].fillna('Unknown')
    enriched_data['department'] = enriched_data['department'].fillna('Unknown')
    enriched_data['hire_date'] = pd.to_datetime(enriched_data['hire_date'], errors='coerce')
    
    # --- CREATE FEATURES (MUST MATCH TRAINING) ---
    # Time features
    enriched_data['hour'] = enriched_data['timestamp_dt'].dt.hour
    enriched_data['minute'] = enriched_data['timestamp_dt'].dt.minute
    enriched_data['day_of_week'] = enriched_data['timestamp_dt'].dt.dayofweek
    enriched_data['day_of_year'] = enriched_data['timestamp_dt'].dt.dayofyear
    enriched_data['week_of_year'] = enriched_data['timestamp_dt'].dt.isocalendar().week
    
    # Employee tenure
    enriched_data['days_since_hire'] = (
        enriched_data['timestamp_dt'] - enriched_data['hire_date']
    ).dt.days
    enriched_data['days_since_hire'] = enriched_data['days_since_hire'].fillna(0)
    
    # Apply saved encodings
    enriched_data['dept_encoded'] = enriched_data['department'].map(
        lambda x: dept_mapping.get(x, -1)  # -1 for unknown departments
    )
    enriched_data['title_encoded'] = enriched_data['job_title'].map(
        lambda x: title_mapping.get(x, -1)  # -1 for unknown titles
    )
    
    # Ensure all features exist
    for feature in features:
        if feature not in enriched_data.columns:
            print(f"[ERROR AD predict_ml] Missing feature: {feature}")
            # Add placeholder
            enriched_data[feature] = 0
    
    X_predict = enriched_data[features]
    
    # Remove rows with NaN values
    valid_mask = X_predict.notna().all(axis=1)
    X_predict_clean = X_predict[valid_mask]
    enriched_data_clean = enriched_data[valid_mask]
    
    if X_predict_clean.empty:
        print(f"[DEBUG AD predict_ml] No valid data for prediction after cleaning.")
        return pd.DataFrame()
    
    # Scale features
    X_scaled = scaler.transform(X_predict_clean)
    
    # Predict
    enriched_data_clean['Anomaly Score'] = model.decision_function(X_scaled)
    enriched_data_clean['Is Anomaly'] = model.predict(X_scaled)
    
    # Filter for anomalies (-1)
    anomalies_df = enriched_data_clean[enriched_data_clean['Is Anomaly'] == -1].copy()
    
    print(f"[DEBUG AD predict_ml] Total records: {len(enriched_data_clean)}, Anomalies: {len(anomalies_df)}")
    
    # Sort by anomaly score (most negative first)
    if not anomalies_df.empty:
        anomalies_df = anomalies_df.sort_values('Anomaly Score')
        
        # Display top anomalies with context
        print(f"[DEBUG AD predict_ml] Top 5 anomalies:")
        for idx, row in anomalies_df.head().iterrows():
            print(f"  Employee {row['employee_id']} - Score: {row['Anomaly Score']:.3f} - "
                  f"Dept: {row.get('department', 'N/A')} - Time: {row['timestamp_dt'].strftime('%H:%M')}")
    
    return anomalies_df