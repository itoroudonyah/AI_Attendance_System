# AI Attendance System

An AI-powered attendance system with facial recognition, role-based access, and anomaly detection for workforce accountability.

## Features
- Facial recognition check-in/check-out
- Role-based dashboard (Admin/Manager/User/Employee)
- Employee management and face enrollment
- Anomaly detection (rule-based, statistical, row-level evaluation)
- Synthetic data generator for testing
- Location tagging on attendance records

## Tech Stack
- Python
- Streamlit
- SQLite
- OpenCV
- scikit-learn
- Plotly

## Project Structure (high level)
- `Home.py`: main entry points
- `pages/`: Streamlit pages (Employee management, Attendance, Anomaly Detection, etc.)
- `database.py`: DB schema and operations
- `anomaly_detection/`: anomaly detection modules
- `employee_photos/`, `encodings/`, `models/`: assets and ML artifacts

## Screenshots
Add these images under a `screenshots/` folder in the repo root and update filenames if needed.

1. Dashboard  
   `screenshots/dashboard.png`
2. Employee Management  
   `screenshots/employee-management.png`
3. Take Attendance  
   `screenshots/take-attendance.png`
4. Attendance Visuals  
   `screenshots/attendance-visuals.png`
5. Anomaly Detection  
   `screenshots/anomaly-detection.png`
6. Anomaly Visuals  
   `screenshots/anomaly-visuals.png`

Example markdown once files exist:
```md
![Dashboard](screenshots/dashboard.png)
```

## Getting Started
1. Create and activate a virtual environment.
2. Install dependencies.
3. Run the Streamlit app.

Example:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run Home.py
```

## Notes
- Local files like `attendance.db`, model files (`*.pkl`, `*.joblib`), and `employee_photos/` should typically be git-ignored.
- For anomaly evaluation, synthetic data can include `is_anomaly_true` labels.

## License
Add a license if you plan to open-source this project.
