# Home.py - Enhanced with User Authentication and Role-Based Access
import streamlit as st
import database as db
import time
from datetime import datetime
import sqlite3
import hashlib
import pandas as pd
import sys
import os
from navigation import render_sidebar, ensure_session, create_session, clear_session, render_page_header, apply_sidebar_style

# IMPORTANT: Disable Streamlit's automatic page discovery
# This prevents pages/ folder from showing in sidebar
st.set_page_config(
    page_title="AI Attendance System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None  # Hide the hamburger menu
)

# Check for navigation from other pages
if 'page' in st.query_params:
    target_page = st.query_params['page']
    st.query_params.clear()  # Clear the parameter
    
    if target_page == 'manage_employees':
        st.switch_page("pages/1_Manage_Employees.py")
    elif target_page == 'my_attendance':
        st.switch_page("pages/My_Attendance.py")
    elif target_page == 'system_settings':
        st.switch_page("pages/System_Settings.py")
    elif target_page == 'take_attendance':
        st.switch_page("pages/Take_Attendance.py")
    elif target_page == 'view_records':
        st.switch_page("pages/View_Records.py")
    elif target_page == 'analytics':
        st.switch_page("pages/Anomaly_Visuals.py")
    # For dashboard, just continue (it's already Home.py)
    
# Add pages directory to path for proper imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize database tables if they don't exist
db.init_db()

# Custom CSS for better styling
st.markdown("""
<style>
    /* Hide Streamlit's default sidebar navigation */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        display: none !important;
    }
    
    .main-container {
        max-width: 340px;
        margin: 0 auto;
        padding: 1.25rem 0.5rem 1.5rem;
    }
    
    .login-card {
        background: white;
        padding: 1.5rem 1.5rem 1.25rem;
        border-radius: 14px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.08);
        border-top: 4px solid #764ba2;
    }
    
    
    .user-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
        margin-left: 10px;
    }
    
    .admin-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .user-badge-regular {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #764ba2;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .stat-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.75rem 0.85rem;
        border-radius: 10px;
        text-align: center;
    }
    .stat-card h3 {
        font-size: 1.4rem;
        margin: 0 0 0.2rem 0;
    }
    .stat-card p {
        font-size: 0.85rem;
        margin: 0;
    }
    
    .restricted-feature {
        opacity: 0.6;
        filter: grayscale(30%);
        position: relative;
    }
    
    .restricted-label {
        position: absolute;
        top: 10px;
        right: 10px;
        background: #ff6b6b;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7em;
        font-weight: bold;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-left: 5px solid #f5576c;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .nav-button {
        width: 100%;
        margin: 0.25rem 0;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s;
        font-weight: 500;
    }
    
    .nav-button:hover {
        transform: translateX(5px);
        background: #f8f9fa;
    }
    
    .nav-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .nav-button-secondary {
        background: #f8f9fa;
        color: #333;
        border: 1px solid #dee2e6;
    }
    
    .quick-action-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .quick-action-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
        cursor: pointer;
        border: 2px solid transparent;
    }
    
    .quick-action-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
        border-color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'employee_name' not in st.session_state:
    st.session_state.employee_name = None

if 'session_id' not in st.session_state:
    st.session_state.session_id = None

ensure_session(timeout_minutes=None)
apply_sidebar_style()

if not st.session_state.authenticated:
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] {
                display: none !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Database connection helper function
def get_db_connection():
    """Create and return a database connection"""
    return sqlite3.connect(db.DATABASE_NAME)

# Check if database has user management tables, create them if not
def initialize_user_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            employee_name TEXT NOT NULL,
            email TEXT,
            role TEXT NOT NULL DEFAULT 'user',
            employee_id TEXT UNIQUE,
            department TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id)
        )
    ''')
    
    # Create login_logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            logout_time TIMESTAMP,
            ip_address TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Check if admin exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    admin_exists = cursor.fetchone()
    
    if not admin_exists:
        # Create default admin (password: admin123 - should be changed on first login)
        default_password = "admin123"
        password_hash = hashlib.sha256(default_password.encode()).hexdigest()
        
        cursor.execute('''
            INSERT INTO users (username, password_hash, employee_name, email, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', password_hash, 'System Administrator', 'admin@company.com', 'admin'))
        
        conn.commit()
        # Don't show info message on every load
        pass
    
    conn.close()

# Initialize user tables
initialize_user_tables()

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    password_hash = hash_password(password)
    
    cursor.execute('''
        SELECT id, username, employee_name, role, employee_id, department 
        FROM users 
        WHERE username = ? AND password_hash = ? AND is_active = 1
    ''', (username, password_hash))
    
    user = cursor.fetchone()
    conn.close()
    
    return user


def get_user_features(role):
    """Return available features based on user role"""
    all_features = {
        "dashboard": {"title": "🏠 Dashboard", "page": "pages/0_Dashboard.py", "icon": "🏠"},
        "take_attendance": {"title": "Take Attendance", "page": "pages/Take_Attendance.py", "icon": "📸"},
        "my_attendance": {"title": "My Attendance", "page": "pages/My_Attendance.py", "icon": "👤"}
    }
    
    # Add manager/admin features
    if role in ('admin', 'manager'):
        all_features.update({
            "view_records": {"title": "View Records", "page": "pages/View_Records.py", "icon": "📊"},
            "anomaly_detection": {"title": "Anomaly Detection", "page": "pages/Anomaly_Detection.py", "icon": "🚨"},
            "anomaly_visuals": {"title": "Anomaly Visuals", "page": "pages/Anomaly_Visuals.py", "icon": "📈"},
        })

    # Admin-only features
    if role == 'admin':
        all_features.update({
            "manage_employees": {"title": "Manage Employees", "page": "pages/1_Manage_Employees.py", "icon": "👥"},
            "system_settings": {"title": "System Settings", "page": "pages/System_Settings.py", "icon": "⚙️"},
        })
    
    return all_features

# Navigation function - FIXED TO HANDLE TAKE ATTENDANCE
def navigate_to_page(page_name):
    """Navigate to a specific page"""
    page_mapping = {
        "🏠 Dashboard": "pages/0_Dashboard.py",
        "👥 Manage Employees": "pages/1_Manage_Employees.py",
        "📸 Take Attendance": "pages/Take_Attendance.py",  # FIXED: Added mapping
        "📊 View Records": "pages/View_Records.py",
        "🚨 Anomaly Detection": "pages/Anomaly_Detection.py",
        "📈 Anomaly Visuals": "pages/Anomaly_Visuals.py",
        "📈 Analytics": "pages/Anomaly_Visuals.py",
        "👤 My Attendance": "pages/My_Attendance.py",
        "⚙️ System Settings": "pages/System_Settings.py"
    }
    
    if page_name in page_mapping:
        try:
            st.switch_page(page_mapping[page_name])
        except Exception as e:
            st.error(f"Error navigating to {page_name}: {str(e)}")
            st.info(f"Please ensure '{page_mapping[page_name]}' exists in your project.")

# LOGIN PAGE (Shown when not authenticated)
if not st.session_state.authenticated:
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown(
            """
            <style>
                div[data-testid="stForm"] {
                    max-width: 280px;
                    margin: 0 auto;
                }
                div[data-testid="stForm"] [data-testid="stTextInput"],
                div[data-testid="stForm"] [data-testid="stCheckbox"] {
                    margin-bottom: 0.35rem;
                }
                div[data-testid="stForm"] .stButton > button {
                    width: 100%;
                    padding: 0.3rem 0.7rem;
                    font-size: 0.82rem;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.25rem;">
            <h1 style="color: #764ba2; font-size: 1.75rem; margin-bottom: 0.25rem;">🔐 AI Attendance System</h1>
            <p style="margin: 0;">Secure Login Required</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Login Form
    with st.container():
        #st.markdown('<div class="login-card">', unsafe_allow_html=True)
        
        # Login Form
        #st.subheader(" ")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            remember_me = st.checkbox("Remember me")
            
            login_submitted = st.form_submit_button("🔐 Login", use_container_width=True)
            
            if login_submitted:
                if username and password:
                    with st.spinner("Authenticating..."):
                        user = authenticate_user(username, password)
                        if user: 
                            st.session_state.authenticated = True
                            st.session_state.user_id = user[0]
                            st.session_state.username = user[1]
                            st.session_state.employee_name = user[2]
                            role = user[3].lower() if isinstance(user[3], str) else user[3]
                            st.session_state.user_role = role
                            st.session_state.employee_id = user[4]
                            st.session_state.department = user[5]
                            create_session(
                                user_id=user[0],
                                username=user[1],
                                role=role,
                                employee_name=user[2],
                                employee_id=user[4],
                                department=user[5],
                                remember_me=bool(remember_me),
                            )
                            
                            # Log login activity
                            conn = get_db_connection()
                            cursor = conn.cursor()
                            cursor.execute('''
                                INSERT INTO login_logs (user_id, username, login_time)
                                VALUES (?, ?, ?)
                            ''', (user[0], user[1], datetime.now()))
                            conn.commit()
                            conn.close()
                            
                            st.success(f"Welcome back, {user[2]}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Invalid username or password!")
                else:
                    st.warning("Please enter both username and password")
        

        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer note
    st.markdown("""
    <div style="text-align: center; margin-top: 1.25rem; color: #666;">
        <p>💼 <strong>AI-Powered Workforce Accountability System</strong></p>
        <p style="font-size: 0.9em;">Secure • Role-Based • Intelligent System</p>
        <p style="font-size: 0.8em;">© 2026 Developed by: Itoro Udonnyah (NOU234244897) | <a href="https://github.com/itoroudonyah" target="_blank">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
else:
    # MAIN DASHBOARD (Shown after authentication)
    # Header with user info
    render_page_header("🤖 AI-Powered Attendance Management System")
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        badge_class = "admin-badge" if st.session_state.user_role == 'admin' else "user-badge-regular"
        user_display = f"{st.session_state.employee_name}"
        if st.session_state.employee_id:
            user_display += f" (ID: {st.session_state.employee_id})"
        
        
        st.markdown(f"""
        <div style="display: flex; align-items: center;">
            <span class="user-badge {badge_class}">{st.session_state.user_role.upper()}</span>
        </div>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Welcome back, <strong>{user_display}</strong>!
            {f'<br><small>Department: {st.session_state.department}</small>' if st.session_state.department else ''}
        </p>
        """, unsafe_allow_html=True)
    
    with col3:
        logout_col_left, logout_col_right = st.columns([1, 1])
        with logout_col_left:
            if st.button("🚪 Logout", use_container_width=True, type="secondary"):
                # Log logout activity
                conn = get_db_connection()
                cursor = conn.cursor()
                # FIXED: Correct SQL syntax
                cursor.execute('''
                    UPDATE login_logs 
                    SET logout_time = ?
                    WHERE id = (
                        SELECT id FROM login_logs 
                        WHERE user_id = ? AND logout_time IS NULL 
                        ORDER BY login_time DESC 
                        LIMIT 1
                    )
                ''', (datetime.now(), st.session_state.user_id))
                conn.commit()
                conn.close()
                
                # Clear session state
                clear_session()
                st.rerun()
    
    # Quick Stats Row (filtered by role)
    st.markdown('<h3 style="color:#6078ea;">📊 Your Dashboard Overview</h3><hr style="border: 1px solid #6078ea;">', unsafe_allow_html=True)
    #st.subheader("📊 :blue[Your Dashboard Overview]", divider="blue")
    
    # Get user-specific stats
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if st.session_state.user_role in ('admin', 'manager'):
        # Admin/manager sees org-wide stats
        cursor.execute("SELECT COUNT(*) FROM employees")
        total_employees = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM attendance WHERE date(date) = date('now')")
        todays_attendance = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT employee_id) FROM attendance WHERE date(date) = date('now')")
        present_today = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
        total_users = cursor.fetchone()[0]
        
        col1, col2, col3, col4 = st.columns(4, gap="small")
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <h3>{total_employees}</h3>
                <p>👥 Total Employees</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <h3>{todays_attendance}</h3>
                <p>📅 Today's Records</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            attendance_rate = (present_today/total_employees*100) if total_employees > 0 else 0
            st.markdown(f"""
            <div class="stat-card">
                <h3>{attendance_rate:.1f}%</h3>
                <p>✅ Today's Rate</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <h3>{total_users}</h3>
                <p>👤 System Users</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        # Regular user sees personal stats
        # Try to get employee_id from user record or use session employee_id
        employee_id = st.session_state.employee_id
        
        if employee_id:
            # Get today's attendance for this employee
            cursor.execute("""
                SELECT COUNT(*) FROM attendance 
                WHERE employee_id = ? AND date(date) = date('now')
            """, (employee_id,))
            today_count = cursor.fetchone()[0]
            
            # Get this month's attendance
            cursor.execute("""
                SELECT COUNT(DISTINCT date) FROM attendance 
                WHERE employee_id = ? 
                AND strftime('%Y-%m', date) = strftime('%Y-%m', 'now')
            """, (employee_id,))
            month_count = cursor.fetchone()[0]
            
            # Get last check-in time
            cursor.execute("""
                SELECT time FROM attendance 
                WHERE employee_id = ? AND date(date) = date('now') AND check_type = 'check_in'
                ORDER BY time DESC LIMIT 1
            """, (employee_id,))
            last_checkin = cursor.fetchone()
            last_checkin_time = last_checkin[0] if last_checkin else "Not checked in"
        else:
            today_count = 0
            month_count = 0
            last_checkin_time = "N/A"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <h3>{month_count}</h3>
                <p>📅 This Month</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <h3>{today_count}</h3>
                <p>✅ Today</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <h3>{last_checkin_time}</h3>
                <p>🕒 Last Check-in</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Today at a glance (admin/manager)
    if st.session_state.user_role in ('admin', 'manager'):
        today = datetime.now().strftime("%Y-%m-%d")
        cursor.execute(
            """
            SELECT COUNT(DISTINCT employee_id)
            FROM attendance
            WHERE date = ? AND check_type = 'check_in' AND time > ?
            """,
            (today, "09:00"),
        )
        late_arrivals = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(DISTINCT employee_id)
            FROM attendance
            WHERE date = ? AND check_type = 'check_in'
            """,
            (today,),
        )
        checked_in = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(DISTINCT employee_id)
            FROM attendance
            WHERE date = ? AND check_type = 'check_out'
            """,
            (today,),
        )
        checked_out = cursor.fetchone()[0]

        missing_checkouts = max(0, checked_in - checked_out)
        absences = max(0, total_employees - checked_in)

        st.markdown("   ")
        st.markdown('<h3 style="color:#6078ea;">🗓️ Today at a Glance</h3><hr style="border: 1px solid #6078ea;">', unsafe_allow_html=True)
        glance_cols = st.columns(4, gap="small")
        with glance_cols[0]:
            st.metric("Checked In", checked_in)
        with glance_cols[1]:
            st.metric("Late Arrivals", late_arrivals)
        with glance_cols[2]:
            st.metric("Missing Check-outs", missing_checkouts)
        with glance_cols[3]:
            st.metric("Absences", absences)

        st.markdown('<h3 style="color:#6078ea;">🚨 Top Anomalies</h3><hr style="border: 1px solid #6078ea;">', unsafe_allow_html=True)
        cursor.execute(
            """
            SELECT employee_name, date, anomaly_type, notes
            FROM anomaly_log
            ORDER BY date DESC, created_at DESC
            LIMIT 5
            """
        )
        anomalies = cursor.fetchall()
        if anomalies:
            anomalies_df = pd.DataFrame(
                anomalies, columns=["Employee", "Date", "Type", "Notes"]
            )
            st.dataframe(anomalies_df, use_container_width=True, hide_index=True)
        else:
            st.info("No recent anomalies logged.")

    conn.close()
    
    user_features = get_user_features(st.session_state.user_role)
    
    # Getting Started Guide
    st.markdown("   ")
    st.markdown("   ")
    st.markdown('<h3 style="color:#6078ea;">🎯 Getting Started</h3><hr style="border: 1px solid #6078ea;">', unsafe_allow_html=True)

    if st.session_state.user_role == 'admin':
        st.markdown("""
        ### 📋 Admin Quick Guide: """)
                    
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            1. **👥 Manage Employees**  
               Start by enrolling employees with photos
            
            2. **📸 Take Attendance**  
               Test the facial recognition system
            
            3. **📊 View Records**  
               Monitor attendance across organization
            """)
        
        with col2:
            st.markdown("""
            4. **📈 Anomaly Visuals**  
               Review anomaly insights and reports
            
            5. **⚙️ System Settings**  
               Configure system parameters
            """)
    elif st.session_state.user_role == 'manager':
        st.markdown("""
        ### 📋 Manager Quick Guide:
        
        1. **📸 Take Attendance**  
           Run attendance sessions for teams
        
        2. **📊 View Records**  
           Review attendance across departments
        
        3. **🚨 Anomaly Detection**  
           Investigate attendance anomalies
        
        4. **📈 Anomaly Visuals**  
           Monitor trends and patterns
        """)
    else:
        st.markdown("""
        ### 📋 User Quick Guide:
        
        1. **📸 Take Attendance**  
           Mark your daily attendance using facial recognition
        
        2. **👤 View My Records**  
           Check your personal attendance history
        
        3. **🏠 Dashboard**  
           Return to main dashboard anytime
        
        **Note:** If you encounter issues, contact your administrator.
        """)
    
    # Footer
    st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
    footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
    with footer_col2:
        st.markdown(f"""
        <div style="text-align: center; color: #666;">
            <p>💼 <strong>AI-Powered Attendance System</strong></p>
            <p style="font-size: 0.9em;">Logged in as: {st.session_state.username} | Role: {st.session_state.user_role}</p>
            <p style="font-size: 0.8em;">Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Version 2.1 | Developed by: Itoro Udonyah (NOU234244897) | <a href="https://github.com/itoroudonyah" target="_blank">GitHub</a></p>
        </div>
        """, unsafe_allow_html=True)

render_sidebar("🏠 Dashboard")
    
