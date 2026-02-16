# pages/5_Anomaly_Detection.py
"""
COMPLETE ANOMALY DETECTION DASHBOARD
Integrated view of Rule-Based, Statistical, and ML Detection methods
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt
import json
import io
import time
import sys
import os
import traceback
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import database as db
from navigation import apply_sidebar_style, render_sidebar, ensure_session, require_roles, render_page_header

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_sidebar_style()
ensure_session(timeout_minutes=None)
render_sidebar("🚨 Anomaly Detection")
require_roles(("admin", "manager"))

# Helper functions
def _classify_ensemble_severity(methods_flagged, ensemble_score):
    """Classify ensemble anomaly severity."""
    if methods_flagged == 3:
        return 'Critical'
    elif methods_flagged == 2:
        return 'High' if ensemble_score > 0.6 else 'Medium'
    elif methods_flagged == 1:
        return 'Medium' if ensemble_score > 0.5 else 'Low'
    else:
        return 'Normal'

def _persist_ensemble_anomalies(results_df: pd.DataFrame) -> None:
    """Write high-confidence ensemble anomalies to anomaly_log for the dashboard."""
    if results_df is None or results_df.empty:
        return

    if "ensemble_severity" not in results_df.columns:
        return

    to_log = results_df[results_df["ensemble_severity"].isin(["High", "Critical"])].copy()
    if to_log.empty:
        return

    to_log["date_str"] = pd.to_datetime(to_log["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    to_log = to_log.dropna(subset=["date_str"])
    if to_log.empty:
        return

    conn = db.get_connection()
    cursor = conn.cursor()

    # Clear prior analysis logs in this date window to avoid duplicates on rerun.
    min_date = to_log["date_str"].min()
    max_date = to_log["date_str"].max()
    cursor.execute(
        """
        DELETE FROM anomaly_log
        WHERE check_type = 'analysis' AND date BETWEEN ? AND ?
        """,
        (min_date, max_date),
    )

    for _, row in to_log.iterrows():
        methods = int(row.get("total_methods_flagged", 0))
        score = float(row.get("ensemble_score", 0.0))
        notes = f"Ensemble {row['ensemble_severity']} ({methods} methods, score={score:.2f})"
        cursor.execute(
            """
            INSERT INTO anomaly_log (employee_id, employee_name, date, check_type, anomaly_type, notes, location_city)
            VALUES (?, ?, ?, 'analysis', ?, ?, NULL)
            """,
            (
                str(row.get("employee_id", "")),
                str(row.get("employee_name", row.get("employee_id", ""))),
                row["date_str"],
                "ensemble_" + str(row["ensemble_severity"]).lower(),
                notes,
            ),
        )

    conn.commit()
    conn.close()

def _build_report_data(report_type: str, ensemble_df: pd.DataFrame, daily_df: pd.DataFrame) -> dict:
    """Create a lightweight report payload for on-screen rendering and PDF export."""
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "report_type": report_type,
        "summary": {},
        "top_departments": [],
        "top_employees": [],
        "notes": [],
    }

    if ensemble_df is None or ensemble_df.empty:
        report["notes"].append("No anomaly results available. Run detection first.")
        return report

    total_emps = ensemble_df["employee_id"].nunique()
    affected_emps = ensemble_df[ensemble_df["total_methods_flagged"] > 0]["employee_id"].nunique()
    total_days = len(ensemble_df)
    anomaly_days = (ensemble_df["total_methods_flagged"] > 0).sum()

    report["summary"] = {
        "total_employees": int(total_emps),
        "affected_employees": int(affected_emps),
        "total_days": int(total_days),
        "anomaly_days": int(anomaly_days),
        "anomaly_rate_pct": round((anomaly_days / total_days * 100), 2) if total_days else 0.0,
    }

    if "department" in ensemble_df.columns:
        dept_counts = (
            ensemble_df[ensemble_df["total_methods_flagged"] > 0]
            .groupby("department")["employee_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(5)
        )
        report["top_departments"] = [
            {"department": str(k), "affected_employees": int(v)} for k, v in dept_counts.items()
        ]

    emp_counts = (
        ensemble_df[ensemble_df["total_methods_flagged"] > 0]
        .groupby("employee_id")["date"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )
    report["top_employees"] = [
        {"employee_id": str(k), "anomaly_days": int(v)} for k, v in emp_counts.items()
    ]

    if daily_df is None or daily_df.empty:
        report["notes"].append("Daily summaries were not available for extended insights.")

    return report

def _report_to_pdf_bytes(report: dict) -> bytes | None:
    """Create a simple PDF from the report. Returns bytes or None if PDF lib unavailable."""
    try:
        from fpdf import FPDF  # type: ignore
    except Exception:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Anomaly Detection Report", ln=True)
    pdf.cell(0, 8, f"Type: {report.get('report_type', 'N/A')}", ln=True)
    pdf.cell(0, 8, f"Generated: {report.get('generated_at', '')}", ln=True)
    pdf.ln(2)

    summary = report.get("summary", {})
    if summary:
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, "Summary", ln=True)
        pdf.set_font("Arial", size=10)
        for k, v in summary.items():
            pdf.cell(0, 7, f"- {k.replace('_', ' ').title()}: {v}", ln=True)
        pdf.ln(1)

    top_depts = report.get("top_departments", [])
    if top_depts:
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, "Top Affected Departments", ln=True)
        pdf.set_font("Arial", size=10)
        for item in top_depts:
            pdf.cell(0, 7, f"- {item['department']}: {item['affected_employees']} employees", ln=True)
        pdf.ln(1)

    top_emps = report.get("top_employees", [])
    if top_emps:
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, "Top Affected Employees", ln=True)
        pdf.set_font("Arial", size=10)
        for item in top_emps:
            pdf.cell(0, 7, f"- {item['employee_id']}: {item['anomaly_days']} anomaly days", ln=True)
        pdf.ln(1)

    notes = report.get("notes", [])
    if notes:
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, "Notes", ln=True)
        pdf.set_font("Arial", size=10)
        for n in notes:
            pdf.multi_cell(0, 6, f"- {n}")

    return pdf.output(dest="S").encode("latin1")

def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame safe for Streamlit Arrow serialization."""
    safe = df.copy()
    for col in ("created_at", "hire_date", "timestamp", "date"):
        if col in safe.columns:
            dt = pd.to_datetime(safe[col], errors="coerce", dayfirst=True, format="mixed")
            # Use consistent strings to avoid mixed-type Arrow issues.
            safe[col] = dt.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    for col in safe.columns:
        if safe[col].dtype == "object":
            safe[col] = safe[col].astype(str)
    return safe

def _safe_dataframe(df: pd.DataFrame, **kwargs) -> None:
    """Render a dataframe and show the exact traceback if it fails."""
    try:
        st.dataframe(_arrow_safe(df), **kwargs)
    except Exception:
        st.error("Dataframe render failed:")
        st.code(traceback.format_exc(), language="text")

def _evaluation_explanation(tp: int, fp: int, fn: int, tn: int, precision: float, recall: float, f1: float, support: int) -> str:
    """Create number-aware interpretation text for evaluation metrics."""
    total_flagged = tp + fp
    total_true = tp + fn
    precision_pct = precision * 100 if precision is not None else 0
    recall_pct = recall * 100 if recall is not None else 0
    f1_pct = f1 * 100 if f1 is not None else 0

    return (
        f"- Precision: {tp} correct alerts out of {total_flagged} flagged → **{precision_pct:.1f}%** of alerts are true anomalies.\n"
        f"- Recall: {tp} detected out of {total_true} true anomalies → **{recall_pct:.1f}%** of real anomalies were caught.\n"
        f"- F1 Score: **{f1_pct:.1f}%** balance between precision and recall (higher is better).\n"
        f"- Support: **{support}** true anomalies in the evaluation set (low support makes scores unstable).\n"
        f"- Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}."
    )

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Try different import methods
try:
    # Method 1: Direct import from anomaly_detection module
    from anomaly_detection import (
        AttendanceDataPreprocessor,
        RuleBasedAnomalyDetector,
        AdaptiveStatisticalDetector,
        create_ml_detector
    )
    print("✓ Imported from anomaly_detection package")
except ImportError as e:
    print(f"Package import failed: {e}")
    try:
        # Method 2: Try importing modules directly
        import importlib.util

        # Import data_preparation
        data_prep_path = os.path.join(parent_dir, "anomaly_detection", "data_preparation.py")
        if os.path.exists(data_prep_path):
            spec = importlib.util.spec_from_file_location("data_preparation", data_prep_path)
            data_prep_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(data_prep_module)
            AttendanceDataPreprocessor = data_prep_module.AttendanceDataPreprocessor

        # Import rule_based_detector
        rule_path = os.path.join(parent_dir, "anomaly_detection", "rule_based_detector.py")
        if os.path.exists(rule_path):
            spec = importlib.util.spec_from_file_location("rule_based_detector", rule_path)
            rule_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rule_module)
            RuleBasedAnomalyDetector = rule_module.RuleBasedAnomalyDetector

        # Import statistical_detector
        stat_path = os.path.join(parent_dir, "anomaly_detection", "statistical_detector.py")
        if os.path.exists(stat_path):
            spec = importlib.util.spec_from_file_location("statistical_detector", stat_path)
            stat_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(stat_module)
            AdaptiveStatisticalDetector = stat_module.AdaptiveStatisticalDetector

        # Import ml_detector
        ml_path = os.path.join(parent_dir, "anomaly_detection", "ml_detector.py")
        if os.path.exists(ml_path):
            spec = importlib.util.spec_from_file_location("ml_detector", ml_path)
            ml_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ml_module)
            create_ml_detector = ml_module.create_ml_detector

        print("✓ Imported modules directly")
    except Exception as e2:
        st.error(f"Failed to import anomaly detection modules: {e2}")
        st.stop()

# Rest of your code continues here...

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main styling */
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        border-left: 5px solid #4CAF50;
        margin-bottom: 1rem;
    }
    
    .kpi-card.critical {
        border-left-color: #f44336;
    }
    
    .kpi-card.warning {
        border-left-color: #ff9800;
    }
    
    .kpi-card.info {
        border-left-color: #2196F3;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-change {
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .positive {
        color: #27ae60;
    }
    
    .negative {
        color: #e74c3c;
    }
    
    .tab-content {
        padding: 1.5rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-top: 1rem;
    }
    
    .method-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e0e6ed;
        transition: transform 0.2s;
    }
    
    .method-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .severity-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .severity-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
    }
    
    .severity-high {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        color: white;
    }
    
    .severity-medium {
        background: linear-gradient(135deg, #42a5f5 0%, #2196F3 100%);
        color: white;
    }
    
    .severity-low {
        background: linear-gradient(135deg, #66bb6a 0%, #4CAF50 100%);
        color: white;
    }
    
    .detection-method-badge {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .method-rule {
        background: #e3f2fd;
        color: #1565c0;
    }
    
    .method-statistical {
        background: #f3e5f5;
        color: #7b1fa2;
    }
    
    .method-ml {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stSelectbox, .stDateInput, .stNumberInput {
        margin-bottom: 1rem;
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'anomaly_data_loaded' not in st.session_state:
    st.session_state.anomaly_data_loaded = False
if 'prepared_data' not in st.session_state:
    st.session_state.prepared_data = None
if 'daily_summaries' not in st.session_state:
    st.session_state.daily_summaries = None
if 'rule_anomalies' not in st.session_state:
    st.session_state.rule_anomalies = None
if 'statistical_anomalies' not in st.session_state:
    st.session_state.statistical_anomalies = None
if 'ml_anomalies' not in st.session_state:
    st.session_state.ml_anomalies = None
if 'ensemble_results' not in st.session_state:
    st.session_state.ensemble_results = None
if "run_detection_requested" not in st.session_state:
    st.session_state.run_detection_requested = False

# Header
render_page_header("🚨 Anomaly Detection")

# <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
#         Integrated detection using Rule-Based, Statistical, and Machine Learning methods
#     </p>
st.text('Integrated detection using Rule-Based, Statistical, and Machine Learning methods')
# Sidebar - Controls
with st.sidebar:
    st.markdown("### 🎛️ Dashboard Controls")
    
    # Data source selection
    data_source = st.radio(
        "Data Source:",
        ["Use loaded data", "Load new data"],
        help="Use previously loaded data or load new attendance data"
    )
    
    if data_source == "Load new data":
        st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
        st.markdown("#### 📂 Data Loading")
        
        # Check if data is in session state from View Records
        if 'current_records_df' in st.session_state and not st.session_state.current_records_df.empty:
            st.info("📊 Data available from View Records page")
            use_existing = st.checkbox("Use existing data", value=True)
            
            if use_existing:
                raw_data = st.session_state.current_records_df
                st.success(f"✅ Loaded {len(raw_data)} records")
            else:
                # File upload option
                uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
                if uploaded_file:
                    raw_data = pd.read_csv(uploaded_file)
                    st.success(f"✅ Uploaded {len(raw_data)} records")
                else:
                    raw_data = None
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file:
                raw_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Uploaded {len(raw_data)} records")
            else:
                raw_data = None
    
    else:
        # Use existing data
        if st.session_state.anomaly_data_loaded and st.session_state.prepared_data is not None:
            st.success("✅ Using previously loaded data")
            raw_data = st.session_state.prepared_data
        else:
            st.warning("No data loaded. Please load data first.")
            raw_data = None
    
    st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
    
    # Date range filter
    st.markdown("#### 📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Employee filter
    st.markdown("#### 👥 Employee Filter")
    show_all = st.checkbox("Show all employees", value=True)
    
    if not show_all and raw_data is not None and 'employee_id' in raw_data.columns:
        employees = sorted(raw_data['employee_id'].unique())
        selected_employees = st.multiselect("Select employees", employees)
    else:
        selected_employees = None
    
    st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
    
    # Detection configuration
    st.markdown("#### ⚙️ Detection Settings")
    
    with st.expander("Rule-Based Settings"):
        lateness_threshold = st.slider("Lateness threshold (minutes)", 15, 120, 90)
        short_day_threshold = st.slider("Short day threshold (hours)", 2, 6, 5)
        max_locations = st.slider("Max locations per day", 1, 5, 2)
    
    with st.expander("Statistical Settings"):
        contamination_rate = st.slider("Contamination rate", 0.01, 0.3, 0.1, 0.01)
        z_threshold = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)
    
    with st.expander("ML Settings"):
        ml_threshold = st.slider("ML probability threshold", 0.1, 1.0, 0.85, 0.05)
        enable_adaptive = st.checkbox("Enable adaptive learning", value=True)
        debug_ml = st.checkbox("Debug ML steps (logs to terminal)", value=False)
        fast_ml = False
        force_full_retrain = st.checkbox("Force retrain now", value=False)
    
    with st.expander("Ensemble Settings"):
        rule_score_threshold = st.slider("Rule severity threshold", 0.0, 3.0, 2.0, 0.1)
        stat_score_threshold = st.slider("Statistical score threshold", 0.0, 1.0, 0.6, 0.05)
        ml_score_threshold = st.slider("ML score threshold", 0.0, 1.0, 0.7, 0.05)
        
        st.markdown("**Method Weights**")
        wcol1, wcol2, wcol3 = st.columns(3)
        with wcol1:
            rule_weight = st.number_input("Rule weight", 0.0, 5.0, 1.0, 0.1)
        with wcol2:
            stat_weight = st.number_input("Statistical weight", 0.0, 5.0, 1.0, 0.1)
        with wcol3:
            ml_weight = st.number_input("ML weight", 0.0, 5.0, 1.0, 0.1)

    st.session_state.lateness_threshold = lateness_threshold
    st.session_state.short_day_threshold = short_day_threshold
    st.session_state.max_locations = max_locations
    st.session_state.contamination_rate = contamination_rate
    st.session_state.z_threshold = z_threshold
    st.session_state.ml_threshold = ml_threshold
    st.session_state.enable_adaptive = enable_adaptive
    st.session_state.debug_ml = debug_ml
    st.session_state.fast_ml = fast_ml
    st.session_state.force_full_retrain = force_full_retrain
    st.session_state.rule_score_threshold = rule_score_threshold
    st.session_state.stat_score_threshold = stat_score_threshold
    st.session_state.ml_score_threshold = ml_score_threshold
    st.session_state.rule_weight = rule_weight
    st.session_state.stat_weight = stat_weight
    st.session_state.ml_weight = ml_weight
    
    st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
    
    # Actions
    st.markdown("#### 🚀 Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        run_detection = st.button("▶️ Run Detection", use_container_width=True)
        if run_detection:
            st.session_state.run_detection_requested = True
    with col2:
        clear_cache = st.button("🗑️ Clear Cache", use_container_width=True)
    
    if clear_cache:
        st.session_state.anomaly_data_loaded = False
        st.session_state.prepared_data = None
        st.session_state.daily_summaries = None
        st.session_state.rule_anomalies = None
        st.session_state.statistical_anomalies = None
        st.session_state.ml_anomalies = None
        st.session_state.ensemble_results = None
        st.rerun()

daily_summaries = None

# Main content area
#if raw_data is not None:
 #   cleaned_data, daily_summaries = preprocessor.prepare_data_pipeline(raw_data)
if raw_data is None and not st.session_state.anomaly_data_loaded:
    st.warning("""
    ## 📊 No Data Loaded
    
    Please load attendance data to begin anomaly detection.
    
    **Options:**
    1. Upload a CSV file using the sidebar
    2. Use data from the View Records page
    3. Connect to system database
    
    Once data is loaded, you can configure detection settings and run the analysis.
    """)
    
    # Quick start guide
    with st.expander("📚 Quick Start Guide"):
        st.markdown("""
        ### How to use this dashboard:
        
        1. **Load Data**: Use the sidebar to upload CSV or use existing data
        2. **Configure Settings**: Adjust detection thresholds as needed
        3. **Run Detection**: Click 'Run Detection' to analyze the data
        4. **Review Results**: Explore anomalies using different detection methods
        5. **Take Action**: Export reports or investigate specific cases
        
        ### Detection Methods:
        
        **🔍 Rule-Based Detection**
        - Uses business rules and thresholds
        - Easy to understand and configure
        - Good for policy violations
        
        **📊 Statistical Detection**
        - Uses statistical methods and outliers
        - Identifies deviations from normal patterns
        - Good for spotting unusual behavior
        
        **🤖 ML Detection (Isolation Forest)**
        - Uses machine learning to detect anomalies
        - Learns patterns from data
        - Good for complex, non-obvious anomalies
        """)
    
    st.stop()
else:
    daily_summaries = st.session_state.daily_summaries.copy() if st.session_state.daily_summaries is not None else None

# Data Preparation Section
st.markdown('<div class="section-header" style="color: #6078ea;">1. Data Preparation & Quality</div>', unsafe_allow_html=True)
if "prep_in_progress" not in st.session_state:
    st.session_state.prep_in_progress = False
if "last_prep_signature" not in st.session_state:
    st.session_state.last_prep_signature = None

# Detect settings/filter changes that should trigger a fresh prep run
settings_signature = (
    st.session_state.get("lateness_threshold"),
    st.session_state.get("short_day_threshold"),
    st.session_state.get("max_locations"),
    st.session_state.get("contamination_rate"),
    st.session_state.get("ml_threshold"),
    st.session_state.get("rule_score_threshold"),
    st.session_state.get("stat_score_threshold"),
    st.session_state.get("ml_score_threshold"),
    st.session_state.get("rule_weight"),
    st.session_state.get("stat_weight"),
    st.session_state.get("ml_weight"),
    st.session_state.get("fast_ml"),
    str(start_date),
    str(end_date),
    tuple(selected_employees) if selected_employees else (),
)
settings_changed = settings_signature != st.session_state.last_prep_signature
if settings_changed:
    st.session_state.last_prep_signature = settings_signature
    st.session_state.prep_in_progress = True
    st.session_state.rule_anomalies = None
    st.session_state.statistical_anomalies = None
    st.session_state.ml_anomalies = None
    st.session_state.ensemble_results = None
    st.session_state.report_data = None

run_detection = st.session_state.run_detection_requested
if raw_data is not None or run_detection:
    # Clear visible outputs only when a fresh detection run is requested
    if run_detection:
        st.session_state.prep_in_progress = True
        st.session_state.rule_anomalies = None
        st.session_state.statistical_anomalies = None
        st.session_state.ml_anomalies = None
        st.session_state.ensemble_results = None
        st.session_state.report_data = None
    elif not st.session_state.prep_in_progress:
        st.session_state.prep_in_progress = False

    with st.spinner("🔄 Preparing data for analysis..."):
        try:
            # Initialize preprocessor
            preprocessor = AttendanceDataPreprocessor(
                standard_start_time="08:00",
                standard_end_time="17:00",
                break_start_time="13:00",
                break_end_time="14:00"
            )
            
            preview = raw_data.head().copy()

            if 'created_at' in preview.columns:
                preview_dt = pd.to_datetime(preview['created_at'], errors='coerce', dayfirst=True, format='mixed')
                preview['created_at'] = preview_dt.dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')

            # Optional: make all remaining object columns safe for Arrow
            for c in preview.columns:
                if preview[c].dtype == "object":
                    preview[c] = preview[c].astype(str)

            #st.dataframe(preview)

            # Prepare data
            if raw_data is not None:
                # Check if data has the right structure
                st.write("🔍 Data structure check:")
                st.write(f"Columns: {list(raw_data.columns)}")
                st.write(f"First few rows:")
                safe_preview = raw_data.head().copy()
                if 'created_at' in safe_preview.columns:
                    created_at_dt = pd.to_datetime(
                        safe_preview['created_at'], errors='coerce', dayfirst=True, format='mixed'
                    )
                    safe_preview['created_at'] = created_at_dt.dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
                for col in safe_preview.columns:
                    if safe_preview[col].dtype == "object":
                        safe_preview[col] = safe_preview[col].astype(str)
                _safe_dataframe(safe_preview)
                
                # Ensure we have the required columns
                required_cols = ['employee_id', 'date', 'time', 'check_type']
                missing_cols = [col for col in required_cols if col not in raw_data.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                    st.info("""
                    Your data should contain these columns:
                    - employee_id (Employee ID)
                    - date (Date of check)
                    - time (Time of check)
                    - check_type (check_in or check_out)
                    
                    Optional columns:
                    - employee_name, department, job_title, ip_address, location_city, etc.
                    """)
                    
                    # Try to find alternative column names
                    st.write("Looking for alternative column names...")
                    col_mapping = {}
                    for req_col in required_cols:
                        for actual_col in raw_data.columns:
                            if req_col.lower() in actual_col.lower():
                                col_mapping[req_col] = actual_col
                                break
                    
                    if col_mapping:
                        st.info(f"Found mappings: {col_mapping}")
                        # Rename columns
                        raw_data = raw_data.rename(columns=col_mapping)
                    else:
                        st.stop()
                
                # Convert date/time to datetime if needed
                if 'date' in raw_data.columns and 'time' in raw_data.columns:
                    combined = raw_data['date'].astype(str) + ' ' + raw_data['time'].astype(str)
                    ts = pd.to_datetime(combined, errors='coerce', dayfirst=True, format='mixed')
                    if ts.notna().any():
                        raw_data['timestamp'] = ts
                    else:
                        ts_date = pd.to_datetime(raw_data['date'], errors='coerce', dayfirst=True, format='mixed')
                        if ts_date.notna().any():
                            raw_data['timestamp'] = ts_date
                        else:
                            st.warning("Could not reliably parse date/time columns; continuing without 'timestamp'.")
                
                # Process the data
                cleaned_data, daily_summaries = preprocessor.prepare_data_pipeline(raw_data)
                
                # Store in session state
                st.session_state.prepared_data = cleaned_data
                st.session_state.daily_summaries = daily_summaries
                st.session_state.anomaly_data_loaded = True
                
                st.success(f"✅ Data prepared successfully!")
                st.write(f"Cleaned records: {len(cleaned_data)}")
                st.write(f"Daily summaries: {len(daily_summaries)}")
                
                # Show sample of prepared data
                # with st.expander("View prepared data sample"):
                #     _safe_dataframe(daily_summaries.head())
                
                # Apply date filter
                if 'date' in daily_summaries.columns:
                    daily_summaries['date'] = pd.to_datetime(
                        daily_summaries['date'], errors='coerce', dayfirst=True, format='mixed'
                    )
                    daily_summaries = daily_summaries.dropna(subset=['date'])
                    mask = (daily_summaries['date'] >= pd.Timestamp(start_date)) & \
                           (daily_summaries['date'] <= pd.Timestamp(end_date))
                    daily_summaries = daily_summaries[mask].copy()
                    st.info(f"Filtered to {len(daily_summaries)} records in date range")
                
                # Apply employee filter
                if selected_employees and len(selected_employees) > 0:
                    daily_summaries = daily_summaries[daily_summaries['employee_id'].isin(selected_employees)].copy()
                    st.info(f"Filtered to {len(daily_summaries)} records for selected employees")
            
            else:
                # Use existing prepared data
                daily_summaries = st.session_state.daily_summaries.copy()
                
                # Apply filters
                if 'date' in daily_summaries.columns:
                    daily_summaries['date'] = pd.to_datetime(
                        daily_summaries['date'], errors='coerce', dayfirst=True, format='mixed'
                    )
                    daily_summaries = daily_summaries.dropna(subset=['date'])
                    mask = (daily_summaries['date'] >= pd.Timestamp(start_date)) & \
                           (daily_summaries['date'] <= pd.Timestamp(end_date))
                    daily_summaries = daily_summaries[mask].copy()
                
                if selected_employees and len(selected_employees) > 0:
                    daily_summaries = daily_summaries[daily_summaries['employee_id'].isin(selected_employees)].copy()
            
            # Display data quality metrics only if we have data
            if daily_summaries is not None and not daily_summaries.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("""
                    <div class="kpi-card">
                        <div class="metric-label">Total Records</div>
                        <div class="metric-value">{:,}</div>
                        <div class="metric-change positive">✓ Ready for analysis</div>
                    </div>
                    """.format(len(daily_summaries)), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="kpi-card">
                        <div class="metric-label">Employees</div>
                        <div class="metric-value">{}</div>
                        <div class="metric-label">Active in period</div>
                    </div>
                    """.format(daily_summaries['employee_id'].nunique()), unsafe_allow_html=True)
                
                with col3:
                    if 'work_duration_hours' in daily_summaries.columns:
                        avg_hours = daily_summaries['work_duration_hours'].mean()
                    else:
                        avg_hours = 0
                    st.markdown("""
                    <div class="kpi-card">
                        <div class="metric-label">Avg Hours/Day</div>
                        <div class="metric-value">{:.1f}</div>
                        <div class="metric-label">Per employee</div>
                    </div>
                    """.format(avg_hours), unsafe_allow_html=True)
                
                with col4:
                    date_range = "N/A"
                    if 'date' in daily_summaries.columns and len(daily_summaries) > 0:
                        min_date = daily_summaries['date'].min().strftime('%Y-%m-%d')
                        max_date = daily_summaries['date'].max().strftime('%Y-%m-%d')
                        date_range = f"{min_date} to {max_date}"
                    
                    st.markdown("""
                    <div class="kpi-card">
                        <div class="metric-label">Date Range</div>
                        <div class="metric-value" style="font-size: 1.5rem;">{}</div>
                        <div class="metric-label">Analysis period</div>
                    </div>
                    """.format(date_range), unsafe_allow_html=True)
                
                # Data quality issues
                with st.expander("📋 Data Quality Report"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Missing values
                        missing_count = daily_summaries.isnull().sum().sum()
                        st.metric("Missing Values", missing_count)
                        
                        # Incomplete pairs
                        if 'has_incomplete_pairs' in daily_summaries.columns:
                            incomplete = daily_summaries['has_incomplete_pairs'].sum()
                            st.metric("Incomplete Pairs", incomplete)
                    
                    with col2:
                        # Weekend data
                        if 'is_weekend' in daily_summaries.columns:
                            weekend_count = daily_summaries['is_weekend'].sum()
                            st.metric("Weekend Records", weekend_count)
                        
                        # Holiday data
                        if 'is_holiday' in daily_summaries.columns:
                            holiday_count = daily_summaries['is_holiday'].sum()
                            st.metric("Holiday Records", holiday_count)
                    
                    # Show sample data
                    st.write("Sample of prepared data:")
                    _safe_dataframe(daily_summaries.head(10), use_container_width=True)
            else:
                st.warning("No data available after preparation. Check your input data.")
        
        except Exception as e:
            st.error(f"❌ Error preparing data: {str(e)}")
            import traceback
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())
            st.stop()
        finally:
            if st.session_state.prep_in_progress:
                st.session_state.prep_in_progress = False

# If preparation is running, hide sections below until it finishes
if st.session_state.prep_in_progress:
    st.info("Preparing data... Please wait.")
    st.stop()

st.markdown('<div class="section-header" style="color: #6078ea;">2. Detection Progress</div>', unsafe_allow_html=True)
button_cols = st.columns([1, 1, 1])
with button_cols[0]:
    st.text('Click on the Run Detection button to analyze the data.')
with button_cols[1]:
    run_detection_main = st.button("▶️ Run Detection", use_container_width=True, key="run_detection_main")

if run_detection_main:
    st.session_state.run_detection_requested = True

run_detection = st.session_state.run_detection_requested

if not st.session_state.anomaly_data_loaded:
    st.info("Load data in section 1 to enable detection.")

# Run detection if requested
if run_detection and st.session_state.anomaly_data_loaded:
    # Clear any previously generated report so it only shows after a new report is generated
    st.session_state.report_data = None
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create tabs for progress tracking
    method_tabs = st.tabs(["🔍 Rule-Based", "📊 Statistical", "🤖 ML", "🎯 Ensemble"])
    
    with method_tabs[0]:
        status_text.text("Running Rule-Based Detection...")
        progress_bar.progress(25)
        
        try:
            # Configure rule detector
            rule_config = {
                'lateness_threshold_minutes': lateness_threshold,
                'short_day_threshold_hours': short_day_threshold,
                'max_locations_per_day': max_locations
            }
            
            rule_detector = RuleBasedAnomalyDetector(rule_config)
            rule_anomalies = rule_detector.detect_anomalies(daily_summaries)
            st.session_state.rule_anomalies = rule_anomalies
            
            if not rule_anomalies.empty:
                unique_days = rule_anomalies.groupby(["employee_id", "date"]).size().shape[0]
                unique_emps = rule_anomalies["employee_id"].nunique()
                st.success(
                    f"✅ Found {len(rule_anomalies)} rule-based anomaly records "
                    f"across {unique_days} days and {unique_emps} employees"
                )
                display_rule_anomalies = rule_anomalies.copy()
                if 'anomalies' in display_rule_anomalies.columns:
                    def _format_anomaly_labels(anomalies):
                        if not isinstance(anomalies, list):
                            return str(anomalies)
                        parts = []
                        for anomaly in anomalies:
                            if isinstance(anomaly, dict):
                                type_part = f"{anomaly.get('type', 'unknown')}:{anomaly.get('subtype', 'unknown')}"
                                severity = anomaly.get('severity')
                                if severity:
                                    type_part = f"{type_part} ({severity})"
                                parts.append(type_part)
                            else:
                                parts.append(str(anomaly))
                        return "; ".join(parts)

                    display_rule_anomalies['anomaly_labels'] = display_rule_anomalies['anomalies'].apply(_format_anomaly_labels)
                    display_rule_anomalies = display_rule_anomalies.drop(columns=['anomalies'])
                
                if 'anomaly_types' in display_rule_anomalies.columns:
                    display_rule_anomalies = display_rule_anomalies.drop(columns=['anomaly_types'])

                _safe_dataframe(display_rule_anomalies, use_container_width=True)
            else:
                st.info("No rule-based anomalies detected")
        
        except Exception as e:
            st.error(f"Rule-based detection error: {e}")
    
    with method_tabs[1]:
        status_text.text("Running Statistical Detection...")
        progress_bar.progress(50)
        
        try:
            # Configure statistical detector
            stat_config = {
                'contamination': contamination_rate,
                'adaptation_rate': 0.1
            }
            
            stat_detector = AdaptiveStatisticalDetector(**stat_config)
            last_stat_contamination = st.session_state.get("last_stat_contamination")
            last_stat_z = st.session_state.get("last_stat_z_threshold")
            
            # Check if training is needed
            if (
                stat_detector.detectors.get('isolation_forest') is None
                or last_stat_contamination != contamination_rate
            ):
                with st.spinner("Training statistical models..."):
                    stat_detector.train_models(daily_summaries)
                st.session_state.last_stat_contamination = contamination_rate
                st.session_state.last_stat_z_threshold = z_threshold
            # if stat_detector.isolation_forest is None:
            #     with st.spinner("Training statistical models..."):
            #         stat_detector.train_models(daily_summaries)
            
            statistical_anomalies = stat_detector.detect_anomalies(
                daily_summaries,
                z_threshold=z_threshold
            )
            st.session_state.statistical_anomalies = statistical_anomalies

            flag_col = "ensemble_anomaly" if "ensemble_anomaly" in statistical_anomalies.columns else "statistical_anomaly"
            anomalies_only = statistical_anomalies[statistical_anomalies[flag_col] == 1]
            
            if not anomalies_only.empty:
                st.success(f"✅ Found {len(anomalies_only)} statistical anomalies")
                _safe_dataframe(anomalies_only, use_container_width=True)
            else:
                st.info("No statistical anomalies detected")
        
        except Exception as e:
            st.error(f"Statistical detection error: {e}")
    
    with method_tabs[2]:
        status_text.text("Running ML Detection...")
        progress_bar.progress(75)
        
        try:
            # Configure ML detector
            ml_detector = create_ml_detector(
                model_dir='models/ml_anomaly',
                contamination=contamination_rate,
                adaptation_rate=0.2
            )
            
            # DEBUG: Check if model exists or retrain if settings changed
            last_ml_contamination = st.session_state.get("last_ml_contamination")
            last_ml_fast = st.session_state.get("last_ml_fast_mode")
            force_full_retrain = st.session_state.get("force_full_retrain", False)
            if (
                ml_detector.isolation_forest is None
                or last_ml_contamination != contamination_rate
                or last_ml_fast != fast_ml
                or force_full_retrain
            ):
                with st.spinner("Training ML model..."):
                    ml_detector.train_isolation_forest(
                        daily_summaries,
                        use_essential_features=fast_ml,
                    )
                st.session_state.last_ml_contamination = contamination_rate
                st.session_state.last_ml_fast_mode = fast_ml
                st.session_state.force_full_retrain = False
            else:
                st.info("✓ Using existing ML model")
            
            ml_anomalies = ml_detector.detect_anomalies(
                daily_summaries,
                probability_threshold=ml_threshold,
                use_advanced_features=True,
                include_contributions=False,
                enable_adaptive=enable_adaptive,
                debug_steps=debug_ml,
                fast_mode=fast_ml,
                sample_size=500 if fast_ml else None,
            )
            st.session_state.ml_anomalies = ml_anomalies

            st.success(f"✅ ML detection complete ({len(ml_anomalies)} records scored)")

            # Status: fast mode + rows + feature count
            feature_count = None
            if hasattr(ml_detector, "feature_columns") and ml_detector.feature_columns is not None:
                feature_count = len(ml_detector.feature_columns)
            elif hasattr(ml_detector, "feature_names") and ml_detector.feature_names is not None:
                feature_count = len(ml_detector.feature_names)
            elif hasattr(ml_detector, "last_feature_columns") and ml_detector.last_feature_columns is not None:
                feature_count = len(ml_detector.last_feature_columns)
            elif hasattr(ml_detector, "feature_count") and ml_detector.feature_count is not None:
                feature_count = ml_detector.feature_count

            st.caption(
                f"Fast mode: {'ON' if fast_ml else 'OFF'} · "
                f"Rows used: {len(ml_anomalies)} · "
                f"Features: {feature_count if feature_count is not None else 'n/a'}"
            )
            
            # DEBUG: Show score distribution
            st.write("📊 ML Score Distribution:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average ML Score", f"{ml_anomalies['ml_anomaly_score'].mean():.3f}")
                st.metric("Min Score", f"{ml_anomalies['ml_anomaly_score'].min():.3f}")
                st.metric("Max Score", f"{ml_anomalies['ml_anomaly_score'].max():.3f}")
            
            with col2:
                # Count by severity
                severity_counts = ml_anomalies['ml_severity'].value_counts()
                st.write("Severity Breakdown:")
                for severity, count in severity_counts.items():
                    st.write(f"{severity}: {count} ({count/len(ml_anomalies):.1%})")
            
            # Show score histogram
            fig = px.histogram(ml_anomalies, x='ml_anomaly_score', 
                            nbins=20, title="ML Anomaly Score Distribution")
            fig.add_vline(x=ml_threshold, line_dash="dash", line_color="red",
                        annotation_text=f"Threshold: {ml_threshold}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top 10 highest scores
            top_scores = ml_anomalies.nlargest(10, 'ml_anomaly_score')[['employee_name', 'date', 'ml_anomaly_score', 'ml_severity']]
            st.write("Top 10 Highest Scores:")
            _safe_dataframe(top_scores, use_container_width=True)
            
            anomalies_only = ml_anomalies[ml_anomalies['ml_anomaly_flag'] == 1]
            
            if not anomalies_only.empty:
                st.success(f"✅ Found {len(anomalies_only)} ML anomalies")
                _safe_dataframe(anomalies_only, use_container_width=True)
            else:
                st.info("No ML anomalies detected")
                
                # DEBUG: Why no anomalies?
                st.write("🔍 Debug: Why no anomalies?")
                st.write(f"- Threshold: {ml_threshold}")
                st.write(f"- Max score: {ml_anomalies['ml_anomaly_score'].max():.3f}")
                st.write(f"- Scores above threshold: {(ml_anomalies['ml_anomaly_score'] >= ml_threshold).sum()}")
                
                if ml_anomalies['ml_anomaly_score'].max() < ml_threshold:
                    st.warning(f"⚠️ Threshold ({ml_threshold}) is higher than max score ({ml_anomalies['ml_anomaly_score'].max():.3f})")
                    st.info("Try lowering the ML probability threshold in the sidebar")

            # Evaluation panel (works when synthetic ground-truth labels exist)
            with st.expander("🧪 Evaluation (Precision/Recall/F1)", expanded=False):
                prepared_df = st.session_state.get("prepared_data")
                if prepared_df is None or prepared_df.empty or "is_anomaly_true" not in prepared_df.columns:
                    st.info("Ground-truth labels not found. Generate/import synthetic data with 'is_anomaly_true'.")
                else:
                    truth_df = prepared_df.copy()
                    truth_df["date"] = pd.to_datetime(truth_df.get("date"), errors="coerce", dayfirst=True, format="mixed")
                    truth_df = truth_df.dropna(subset=["date"])
                    truth_df["date_key"] = truth_df["date"].dt.strftime("%Y-%m-%d")
                    truth_daily = (
                        truth_df.groupby(["employee_id", "date_key"], as_index=False)["is_anomaly_true"]
                        .max()
                    )
                    eval_df = ml_anomalies.copy()
                    eval_df["date_key"] = pd.to_datetime(
                        eval_df["date"], errors="coerce", dayfirst=True, format="mixed"
                    ).dt.strftime("%Y-%m-%d")
                    eval_df = eval_df.merge(
                        truth_daily, on=["employee_id", "date_key"], how="left"
                    )
                    eval_df["is_anomaly_true"] = eval_df["is_anomaly_true"].fillna(0).astype(int)
                    y_true = eval_df["is_anomaly_true"].astype(int)
                    y_pred = eval_df["ml_anomaly_flag"].astype(int)

                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average="binary", zero_division=0
                    )
                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
                    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (0, 0, 0, 0))

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Precision", f"{precision:.3f}")
                    m2.metric("Recall", f"{recall:.3f}")
                    m3.metric("F1 Score", f"{f1:.3f}")
                    m4.metric("Support (Anomalies)", int(y_true.sum()))

                    # Confusion matrix heatmap
                    cm_display = pd.DataFrame(
                        [[tn, fp], [fn, tp]],
                        index=["Actual Normal", "Actual Anomaly"],
                        columns=["Pred Normal", "Pred Anomaly"],
                    )
                    fig_cm = go.Figure(
                        data=go.Heatmap(
                            z=cm_display.values,
                            x=cm_display.columns.tolist(),
                            y=cm_display.index.tolist(),
                            colorscale="Blues",
                            text=cm_display.values,
                            texttemplate="%{text}",
                            textfont={"size": 14},
                        )
                    )
                    fig_cm.update_layout(
                        title="Confusion Matrix",
                        height=320,
                        margin=dict(l=0, r=0, t=40, b=0),
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                    st.markdown("**Interpretation (based on these exact numbers):**")
                    st.markdown(
                        _evaluation_explanation(tp, fp, fn, tn, precision, recall, f1, int(y_true.sum()))
                    )

                    _safe_dataframe(
                        eval_df[
                            [
                                "employee_id",
                                "employee_name",
                                "date",
                                "ml_anomaly_score",
                                "ml_anomaly_flag",
                                "is_anomaly_true",
                            ]
                        ].head(20),
                        use_container_width=True,
                    )
        
        except Exception as e:
            st.error(f"ML detection error: {e}")
            st.code(traceback.format_exc(), language="text")
    
    with method_tabs[3]:
        status_text.text("Creating Ensemble Results...")
        progress_bar.progress(100)
        
        try:
            # Combine results from all methods
            ensemble_results = []
            
            # Prepare base DataFrame with all dates and employees
            all_dates = daily_summaries[['employee_id', 'employee_name', 'department', 'date']].copy()
            
            # DEBUG: Check what data we have
            st.write(f"📊 Base dataset: {len(all_dates)} records")
            
            # Merge rule-based anomalies
            rule_anomalies_count = 0
            if st.session_state.rule_anomalies is not None and not st.session_state.rule_anomalies.empty:
                st.write(f"📋 Rule anomalies: {len(st.session_state.rule_anomalies)} records")
                rule_summary = st.session_state.rule_anomalies.groupby(['employee_id', 'date']).agg({
                    'anomaly_count': 'sum',
                    'severity_score': 'max'
                }).reset_index()
                rule_summary.columns = ['employee_id', 'date', 'rule_anomaly_count', 'rule_severity']
                all_dates = pd.merge(all_dates, rule_summary, on=['employee_id', 'date'], how='left')
                rule_anomalies_count = (rule_summary['rule_anomaly_count'] > 0).sum()
                st.write(f"📋 Rule anomalies (grouped): {rule_anomalies_count} days")
            else:
                st.write("📋 No rule anomalies")
                all_dates['rule_anomaly_count'] = 0
                all_dates['rule_severity'] = 0.0
            
            # Merge statistical anomalies
            stat_anomalies_count = 0
            if st.session_state.statistical_anomalies is not None and not st.session_state.statistical_anomalies.empty:
                st.write(f"📈 Statistical anomalies: {len(st.session_state.statistical_anomalies)} records")
                flag_col = "ensemble_anomaly" if "ensemble_anomaly" in st.session_state.statistical_anomalies.columns else "statistical_anomaly"
                stat_summary = st.session_state.statistical_anomalies.groupby(['employee_id', 'date']).agg({
                    'ensemble_score': 'max',
                    flag_col: 'max'
                }).reset_index()
                stat_summary.columns = ['employee_id', 'date', 'stat_score', 'stat_anomaly']
                all_dates = pd.merge(all_dates, stat_summary, on=['employee_id', 'date'], how='left')
                stat_anomalies_count = (stat_summary['stat_anomaly'] == 1).sum()
                st.write(f"📈 Statistical anomalies (grouped): {stat_anomalies_count} days")
            else:
                st.write("📈 No statistical anomalies")
                all_dates['stat_score'] = 0.0
                all_dates['stat_anomaly'] = 0
            
            # Merge ML anomalies
            ml_anomalies_count = 0
            if st.session_state.ml_anomalies is not None and not st.session_state.ml_anomalies.empty:
                st.write(f"🤖 ML anomalies: {len(st.session_state.ml_anomalies)} records")
                ml_summary = st.session_state.ml_anomalies[['employee_id', 'date', 'ml_anomaly_score', 'ml_anomaly_flag']].copy()
                ml_summary.columns = ['employee_id', 'date', 'ml_score', 'ml_anomaly']
                all_dates = pd.merge(all_dates, ml_summary, on=['employee_id', 'date'], how='left')
                ml_anomalies_count = (ml_summary['ml_anomaly'] == 1).sum()
                st.write(f"🤖 ML anomalies (grouped): {ml_anomalies_count} days")
            else:
                st.write("🤖 No ML anomalies")
                all_dates['ml_score'] = 0.0
                all_dates['ml_anomaly'] = 0
            
            # Fill NaN values
            for col in ['rule_anomaly_count', 'rule_severity', 'stat_score', 'stat_anomaly', 'ml_score', 'ml_anomaly']:
                if col in all_dates.columns:
                    all_dates[col] = all_dates[col].fillna(0)
            
            # DEBUG: Show sample of merged data
            st.write("📋 Sample of merged data:")
            _safe_dataframe(all_dates.head(), use_container_width=True)
            
            # Calculate ensemble metrics
            all_dates['total_methods_flagged'] = 0
            all_dates['ensemble_score'] = 0.0
            total_weight = 0.0
            
            # Rule-based
            if 'rule_severity' in all_dates.columns:
                rule_threshold = st.session_state.get('rule_score_threshold', 1.0)
                all_dates['rule_flag'] = (all_dates['rule_severity'] >= rule_threshold).astype(int)
                all_dates['total_methods_flagged'] += all_dates['rule_flag']
                rule_score_norm = (all_dates['rule_severity'] / 3.0).clip(0, 1)
                rule_weight = st.session_state.get('rule_weight', 1.0)
                all_dates['ensemble_score'] += rule_score_norm * rule_weight
                total_weight += rule_weight
                st.write(f"📋 Rule-based: {all_dates['rule_flag'].sum()} flagged (threshold: {rule_threshold})")
            
            # Statistical
            if 'stat_anomaly' in all_dates.columns:
                stat_threshold = st.session_state.get('stat_score_threshold', 0.6)
                stat_flag = ((all_dates['stat_anomaly'] == 1) & (all_dates['stat_score'] >= stat_threshold)).astype(int)
                all_dates['stat_flag'] = stat_flag
                all_dates['total_methods_flagged'] += stat_flag
                stat_weight = st.session_state.get('stat_weight', 1.0)
                all_dates['ensemble_score'] += all_dates['stat_score'] * stat_weight
                total_weight += stat_weight
                st.write(f"📈 Statistical: {stat_flag.sum()} flagged (threshold: {stat_threshold})")
            
            # ML
            if 'ml_anomaly' in all_dates.columns:
                ml_threshold = st.session_state.get('ml_score_threshold', 0.7)
                ml_flag = ((all_dates['ml_anomaly'] == 1) & (all_dates['ml_score'] >= ml_threshold)).astype(int)
                all_dates['ml_flag'] = ml_flag
                all_dates['total_methods_flagged'] += ml_flag
                ml_weight = st.session_state.get('ml_weight', 1.0)
                all_dates['ensemble_score'] += all_dates['ml_score'] * ml_weight
                total_weight += ml_weight
                st.write(f"🤖 ML: {ml_flag.sum()} flagged (threshold: {ml_threshold})")

            if total_weight > 0:
                all_dates['ensemble_score'] = all_dates['ensemble_score'] / total_weight
            
            # Classify based on ensemble
            all_dates['ensemble_severity'] = all_dates.apply(
                lambda row: _classify_ensemble_severity(row['total_methods_flagged'], row['ensemble_score']),
                axis=1
            )
            
            st.session_state.ensemble_results = all_dates
            _persist_ensemble_anomalies(all_dates)
            st.success(f"✅ Ensemble analysis complete")
            
            # Show summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_anomalies = (all_dates['total_methods_flagged'] > 0).sum()
                st.metric("Total Anomaly Days", total_anomalies)
            
            with col2:
                high_confidence = (all_dates['total_methods_flagged'] >= 2).sum()
                st.metric("High Confidence", high_confidence)
            
            with col3:
                affected_employees = all_dates[all_dates['total_methods_flagged'] > 0]['employee_id'].nunique()
                st.metric("Affected Employees", affected_employees)
            
            # DEBUG: Show distribution of methods flagged
            st.write("📊 Methods Flagged Distribution:")
            methods_dist = all_dates['total_methods_flagged'].value_counts().sort_index()
            for methods, count in methods_dist.items():
                st.write(f"{methods} methods: {count} days ({count/len(all_dates):.1%})")
            
            # Show top ensemble scores
            if total_anomalies > 0:
                top_ensemble = all_dates.nlargest(10, 'ensemble_score')[['employee_name', 'date', 'total_methods_flagged', 'ensemble_score', 'ensemble_severity']]
                st.write("🏆 Top 10 Ensemble Scores:")
                _safe_dataframe(top_ensemble, use_container_width=True)
            else:
                st.info("No anomalies detected by any method")
                
                # Show why
                st.write("🔍 Debug: Why no ensemble anomalies?")
                st.write(f"- Rule-based threshold: {st.session_state.get('rule_score_threshold', 1.0)}")
                st.write(f"- Statistical threshold: {st.session_state.get('stat_score_threshold', 0.6)}")
                st.write(f"- ML threshold: {st.session_state.get('ml_score_threshold', 0.7)}")
                st.write(f"- Rule anomalies found: {rule_anomalies_count}")
                st.write(f"- Statistical anomalies found: {stat_anomalies_count}")
                st.write(f"- ML anomalies found: {ml_anomalies_count}")
                
                # Suggest adjustments
                if rule_anomalies_count == 0 and stat_anomalies_count == 0 and ml_anomalies_count == 0:
                    st.warning("⚠️ No anomalies found by any individual method.")
                    st.info("Try lowering the detection thresholds in the sidebar settings.")
            
        except Exception as e:
            st.error(f"Ensemble analysis error: {e}")
        
        status_text.text("✅ Detection complete!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        st.session_state.run_detection_requested = False

# Show last detection results on rerun when not actively running
if (not run_detection) and st.session_state.anomaly_data_loaded:
    has_any_results = any(
        [
            st.session_state.rule_anomalies is not None and not st.session_state.rule_anomalies.empty,
            st.session_state.statistical_anomalies is not None and not st.session_state.statistical_anomalies.empty,
            st.session_state.ml_anomalies is not None and not st.session_state.ml_anomalies.empty,
            st.session_state.ensemble_results is not None and not st.session_state.ensemble_results.empty,
        ]
    )
    if has_any_results:
        st.info("Showing last detection results.")
        method_tabs = st.tabs(["🔍 Rule-Based", "📊 Statistical", "🤖 ML", "🎯 Ensemble"])

        with method_tabs[0]:
            if st.session_state.rule_anomalies is not None and not st.session_state.rule_anomalies.empty:
                st.success(f"✅ Found {len(st.session_state.rule_anomalies)} rule-based anomalies")
                _safe_dataframe(st.session_state.rule_anomalies, use_container_width=True)
            else:
                st.info("No rule-based anomalies.")

        with method_tabs[1]:
            if st.session_state.statistical_anomalies is not None and not st.session_state.statistical_anomalies.empty:
                st.success(f"✅ Found {len(st.session_state.statistical_anomalies)} statistical anomalies")
                _safe_dataframe(st.session_state.statistical_anomalies, use_container_width=True)
            else:
                st.info("No statistical anomalies.")

        with method_tabs[2]:
            if st.session_state.ml_anomalies is not None and not st.session_state.ml_anomalies.empty:
                st.success(f"✅ ML detection results: {len(st.session_state.ml_anomalies)} rows")
                ml_anomalies = st.session_state.ml_anomalies
                st.caption(f"Fast mode: {'ON' if st.session_state.get('ml_fast_mode') else 'OFF'} · Rows used: {len(ml_anomalies)}")

                score_col = "ml_anomaly_score"
                if score_col in ml_anomalies.columns:
                    st.markdown("📊 ML Score Distribution:")
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("Average ML Score", f"{ml_anomalies[score_col].mean():.3f}")
                    with m2:
                        st.metric("Min Score", f"{ml_anomalies[score_col].min():.3f}")
                    with m3:
                        st.metric("Max Score", f"{ml_anomalies[score_col].max():.3f}")

                if "ml_severity" in ml_anomalies.columns:
                    st.markdown("Severity Breakdown:")
                    severity_counts = ml_anomalies["ml_severity"].value_counts()
                    for severity, count in severity_counts.items():
                        st.write(f"{severity}: {count} ({count/len(ml_anomalies):.1%})")

                if "ml_anomaly_flag" in ml_anomalies.columns:
                    anomalies_only = ml_anomalies[ml_anomalies["ml_anomaly_flag"] == 1]
                    st.success(f"✅ Found {len(anomalies_only)} ML anomalies")

                if score_col in ml_anomalies.columns:
                    st.markdown("Top 5 Highest Scores:")
                    top_scores = ml_anomalies.nlargest(5, score_col)[
                        [c for c in ["employee_name", "date", score_col, "ml_severity"] if c in ml_anomalies.columns]
                    ]
                    _safe_dataframe(top_scores, use_container_width=True)

                _safe_dataframe(ml_anomalies, use_container_width=True)
            else:
                st.info("No ML anomalies.")

        with method_tabs[3]:
            if st.session_state.ensemble_results is not None and not st.session_state.ensemble_results.empty:
                ensemble_df = st.session_state.ensemble_results
                st.success(f"✅ Ensemble results: {len(ensemble_df)} rows")

                total_anomalies = (ensemble_df['total_methods_flagged'] > 0).sum() if 'total_methods_flagged' in ensemble_df.columns else 0
                high_confidence = (ensemble_df['total_methods_flagged'] >= 2).sum() if 'total_methods_flagged' in ensemble_df.columns else 0
                affected_employees = ensemble_df[ensemble_df['total_methods_flagged'] > 0]['employee_id'].nunique() if 'employee_id' in ensemble_df.columns and 'total_methods_flagged' in ensemble_df.columns else 0

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Total Anomaly Days", total_anomalies)
                with c2:
                    st.metric("High Confidence", high_confidence)
                with c3:
                    st.metric("Affected Employees", affected_employees)

                if 'total_methods_flagged' in ensemble_df.columns:
                    st.markdown("📊 Methods Flagged Distribution:")
                    methods_dist = ensemble_df['total_methods_flagged'].value_counts().sort_index()
                    for methods, count in methods_dist.items():
                        st.write(f"{methods} methods: {count} days ({count/len(ensemble_df):.1%})")

                if 'ensemble_score' in ensemble_df.columns:
                    st.markdown("🏆 Top 10 Ensemble Scores:")
                    cols = [c for c in ['employee_name', 'date', 'total_methods_flagged', 'ensemble_score', 'ensemble_severity'] if c in ensemble_df.columns]
                    top_ensemble = ensemble_df.nlargest(10, 'ensemble_score')[cols]
                    _safe_dataframe(top_ensemble, use_container_width=True)

                _safe_dataframe(ensemble_df, use_container_width=True)
            else:
                st.info("No ensemble results.")

# Display results if available
if st.session_state.anomaly_data_loaded:
    
    st.markdown('<div class="section-header" style="color: #6078ea;">3. Executive Overview</div>', unsafe_allow_html=True)
    
    # Create overview metrics
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    
    with overview_col1:
        total_days = len(daily_summaries) if daily_summaries is not None else 0
        anomaly_days = 0
        if st.session_state.ensemble_results is not None:
            anomaly_days = (st.session_state.ensemble_results['total_methods_flagged'] > 0).sum()
        
        anomaly_rate = (anomaly_days / total_days * 100) if total_days > 0 else 0
        
        st.markdown(f"""
        <div class="kpi-card {'critical' if anomaly_rate > 10 else 'warning' if anomaly_rate > 5 else 'info'}">
            <div class="metric-label">Anomaly Rate</div>
            <div class="metric-value">{anomaly_rate:.1f}%</div>
            <div class="metric-label">{anomaly_days} of {total_days} days</div>
        </div>
        """, unsafe_allow_html=True)
    
    with overview_col2:
        if st.session_state.ensemble_results is not None:
            high_risk = (st.session_state.ensemble_results['total_methods_flagged'] >= 2).sum()
        else:
            high_risk = 0
        
        st.markdown(f"""
        <div class="kpi-card {'critical' if high_risk > 5 else 'warning' if high_risk > 0 else 'info'}">
            <div class="metric-label">High-Risk Days</div>
            <div class="metric-value">{high_risk}</div>
            <div class="metric-label">Multiple methods agree</div>
        </div>
        """, unsafe_allow_html=True)
    
    with overview_col3:
        if st.session_state.ensemble_results is not None:
            # "With anomalies" should reflect any flagged anomaly (not just high/critical).
            affected_emps = st.session_state.ensemble_results[
                st.session_state.ensemble_results['total_methods_flagged'] > 0
            ]['employee_id'].nunique()
            if daily_summaries is not None and not daily_summaries.empty and 'employee_id' in daily_summaries.columns:
                total_emps = daily_summaries['employee_id'].nunique()
            else:
                total_emps = st.session_state.ensemble_results['employee_id'].nunique()
        else: 
            affected_emps = 0
            if daily_summaries is not None and not daily_summaries.empty and 'employee_id' in daily_summaries.columns:
                total_emps = daily_summaries['employee_id'].nunique()
            else:
                total_emps = 0
                #st.warning("No employee data available in daily summaries")
        
        st.markdown(f"""
        <div class="kpi-card {'warning' if affected_emps > total_emps * 0.3 else 'info'}">
            <div class="metric-label">Affected Employees</div>
            <div class="metric-value">{affected_emps}/{total_emps}</div>
            <div class="metric-label">With anomalies</div>
        </div>
        """, unsafe_allow_html=True)
    
    with overview_col4:
        # Most common anomaly type
        if st.session_state.rule_anomalies is not None and not st.session_state.rule_anomalies.empty:
            # Extract top anomaly type from rule-based
            all_anomalies = []
            for anomalies_list in st.session_state.rule_anomalies['anomalies']:
                for anomaly in anomalies_list:
                    if 'subtype' in anomaly:
                        all_anomalies.append(anomaly['subtype'])
            
            if all_anomalies:
                from collections import Counter
                top_anomaly = Counter(all_anomalies).most_common(1)[0][0]
                top_anomaly_display = top_anomaly.replace('_', ' ').title()
            else:
                top_anomaly_display = "No anomalies"
        else:
            top_anomaly_display = "No anomalies" if st.session_state.anomaly_data_loaded else "Run detection"
        
        st.markdown(f"""
        <div class="kpi-card info">
            <div class="metric-label">Top Anomaly Type</div>
            <div class="metric-value" style="font-size: 1.2rem;">{top_anomaly_display}</div>
            <div class="metric-label">Most frequent issue</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Method Agreement Matrix
    st.markdown("##### 🎯 Method Agreement Matrix")
    
    if st.session_state.ensemble_results is not None:
        # Create agreement matrix
        agreement_data = []
        methods = ['Rule-Based', 'Statistical', 'ML']
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:  # Only upper triangle
                    # Count agreements (both methods flag anomaly)
                    if method1 == 'Rule-Based':
                        col1 = 'rule_flag' if 'rule_flag' in st.session_state.ensemble_results.columns else None
                    elif method1 == 'Statistical':
                        col1 = 'stat_flag' if 'stat_flag' in st.session_state.ensemble_results.columns else None
                    else:  # ML
                        col1 = 'ml_flag' if 'ml_flag' in st.session_state.ensemble_results.columns else None
                    
                    if method2 == 'Rule-Based':
                        col2 = 'rule_flag' if 'rule_flag' in st.session_state.ensemble_results.columns else None
                    elif method2 == 'Statistical':
                        col2 = 'stat_flag' if 'stat_flag' in st.session_state.ensemble_results.columns else None
                    else:  # ML
                        col2 = 'ml_flag' if 'ml_flag' in st.session_state.ensemble_results.columns else None
                    
                    if col1 and col2:
                        agreement = ((st.session_state.ensemble_results[col1] == 1) & 
                                    (st.session_state.ensemble_results[col2] == 1)).sum()
                        total_both = max(1, (st.session_state.ensemble_results[col1] == 1).sum() + 
                                        (st.session_state.ensemble_results[col2] == 1).sum())
                        
                        agreement_rate = agreement / total_both * 100
                        
                        agreement_data.append({
                            'Method 1': method1,
                            'Method 2': method2,
                            'Agreement': agreement,
                            'Agreement Rate': agreement_rate
                        })
        
        if agreement_data:
            agreement_df = pd.DataFrame(agreement_data)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=agreement_df['Agreement Rate'].values.reshape(-1, 1),
                x=['Agreement Rate'],
                y=[f"{row['Method 1']} vs {row['Method 2']}" for _, row in agreement_df.iterrows()],
                colorscale='RdYlGn_r',
                zmin=0,
                zmax=100,
                text=agreement_df['Agreement Rate'].apply(lambda x: f"{x:.1f}%"),
                texttemplate='%{text}',
                textfont={"size": 14},
                hovertemplate='<b>%{y}</b><br>Agreement: %{z:.1f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title="Method Agreement Rates",
                height=200,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run detection to see method agreement analysis")
    else:
        st.info("Run detection to see method agreement analysis")
    
    # Action Center
    st.markdown('<div class="section-header" style="color: #6078ea;">4. Action Center</div>', unsafe_allow_html=True)

    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        st.markdown("#### 📤 Export Results")
        has_results = (
            st.session_state.ensemble_results is not None
            and not st.session_state.ensemble_results.empty
        )
        if not has_results:
            st.warning("No results to export")

        if "export_notice" not in st.session_state:
            st.session_state.export_notice = False

        def _mark_exported():
            st.session_state.export_notice = True

        st.download_button(
            label="📊 Export to CSV",
            data=st.session_state.ensemble_results.to_csv(index=False) if has_results else "",
            file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not has_results,
            on_click=_mark_exported if has_results else None,
        )

        # if st.session_state.export_notice:
        #     st.success("Download ready. Your results are still loaded below.")
    
    with action_col2:
        st.markdown("#### 🔔 Alert Configuration")
        
        with st.expander("Configure Alerts"):
            if "alert_settings" not in st.session_state:
                st.session_state.alert_settings = {
                    "threshold": 2,
                    "severity": ["Critical", "High"],
                    "updated_at": None,
                }

            alert_threshold = st.slider(
                "Alert threshold (methods agreeing)",
                1,
                3,
                st.session_state.alert_settings.get("threshold", 2),
            )
            alert_severity = st.multiselect(
                "Severity levels to alert",
                ["Critical", "High", "Medium", "Low"],
                default=st.session_state.alert_settings.get("severity", ["Critical", "High"]),
            )

            if st.button("💾 Save Alert Settings", use_container_width=True):
                st.session_state.alert_settings = {
                    "threshold": int(alert_threshold),
                    "severity": list(alert_severity),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.success("Alert settings saved!")

            saved = st.session_state.alert_settings
            st.caption(
                f"Saved: threshold ≥ {saved.get('threshold')} methods, "
                f"severity {', '.join(saved.get('severity', [])) or 'None'}"
            )
            if saved.get("updated_at"):
                st.caption(f"Last updated: {saved['updated_at']}")
    
    with action_col3:
        st.markdown("#### 📋 Generate Report")
        
        if "report_data" not in st.session_state:
            st.session_state.report_data = None

        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Detailed Analysis", "Employee Focused"],
        )

        has_results = (
            st.session_state.ensemble_results is not None
            and not st.session_state.ensemble_results.empty
        )

        if st.button("📄 Generate Report", use_container_width=True):
            if not has_results:
                st.warning("Run detection first to generate a report.")
            else:
                with st.spinner("Generating report..."):
                    st.session_state.report_data = _build_report_data(
                        report_type,
                        st.session_state.ensemble_results,
                        st.session_state.daily_summaries,
                    )
                st.success(f"{report_type} report generated!")

        if st.session_state.report_data:
            report = st.session_state.report_data
            summary = report.get("summary", {})
            pdf_bytes = _report_to_pdf_bytes(report)

# Render generated report after the Action Center columns
if st.session_state.report_data:
    report = st.session_state.report_data
    summary = report.get("summary", {})

    st.markdown('<hr style="border: none; border-top: 1px solid #e1e5ea; margin: 10px 0;">', unsafe_allow_html=True)
    st.markdown("### 📋 Generated Report")
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.markdown(
            f"- Generated: `{report.get('generated_at', '')}`\n"
            f"- Type: `{report.get('report_type', '')}`\n"
            f"- Affected Employees: `{summary.get('affected_employees', 0)}` / `{summary.get('total_employees', 0)}`\n"
            f"- Anomaly Days: `{summary.get('anomaly_days', 0)}` / `{summary.get('total_days', 0)}`\n"
            f"- Anomaly Rate: `{summary.get('anomaly_rate_pct', 0)}%`"
        )
    with right_col:
        summary_table = pd.DataFrame(
            [
                {"Metric": "Affected Employees", "Value": f"{summary.get('affected_employees', 0)} / {summary.get('total_employees', 0)}"},
                {"Metric": "Anomaly Days", "Value": f"{summary.get('anomaly_days', 0)} / {summary.get('total_days', 0)}"},
                {"Metric": "Anomaly Rate", "Value": f"{summary.get('anomaly_rate_pct', 0)}%"},
            ]
        )
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

    if report.get("top_departments"):
        st.markdown("**Top Affected Departments**")
        st.dataframe(pd.DataFrame(report["top_departments"]), use_container_width=True, hide_index=True)

    if report.get("top_employees"):
        st.markdown("**Top Affected Employees**")
        st.dataframe(pd.DataFrame(report["top_employees"]), use_container_width=True, hide_index=True)

    if report.get("notes"):
        st.markdown("**Notes**")
        for note in report["notes"]:
            st.write(f"- {note}")

    if pdf_bytes is None:
        st.warning("PDF export requires the `fpdf` package.")
        st.download_button(
            label="Download PDF",
            data=b"",
            file_name="anomaly_report.pdf",
            mime="application/pdf",
            disabled=True,
            use_container_width=True,
        )
    else:
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# Footer
st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
    <p>AI-Powered Attendance System</p>
    <p>Detection Methods: Rule-Based • Statistical • Machine Learning (Isolation Forest)</p>
             <p style="font-size: 0.8em;">Version 2.1 | Developed by: Itoro Udonyah (NOU234244897) | <a href="https://github.com/itoroudonyah" target="_blank">GitHub</a></p>
</div>
""", unsafe_allow_html=True)
