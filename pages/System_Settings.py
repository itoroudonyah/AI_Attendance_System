# System_Settings.py - System configuration page for admins
import streamlit as st
from datetime import datetime, time
from navigation import apply_sidebar_style, render_sidebar, ensure_session, require_roles, render_page_header

st.set_page_config(
    page_title="System Settings",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_sidebar_style()
ensure_session(timeout_minutes=None)
render_sidebar("⚙️ System Settings")
require_roles(("admin",))

# Database connection
def get_db_connection():
    import database as db
    return db.get_connection()

render_page_header("⚙️ System Settings")
st.subheader("Configure System Parameters and Preferences")

st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)

# Settings tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["🔧 General", "📅 Attendance", "🔐 Security", "📊 Analytics", "🛠️ System Maintenance", "⚠️ Danger Zone"]
)

with tab1:
    st.markdown("### General System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        system_name = st.text_input("System Name", value="AI Attendance System", 
                                   help="Display name for the system")
        company_name = st.text_input("Company Name", value="Your Company", 
                                    help="Organization name")
        timezone = st.selectbox("Timezone", ["UTC", "Africa/Lagos", "America/New_York", 
                                           "Europe/London", "Asia/Tokyo"], index=1)
    
    with col2:
        date_format = st.selectbox("Date Format", ["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"], index=0)
        time_format = st.selectbox("Time Format", ["24-hour", "12-hour"], index=0)
        language = st.selectbox("Language", ["English", "Spanish", "French"], index=0)
    
    # Save button
    if st.button("💾 Save General Settings", type="primary", key="save_general"):
        st.success("General settings saved successfully!" \
        "Pending Actual Implementation")
        # Note: In a real app, you would save these to a settings table

with tab2:
    st.markdown("### Attendance Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⏰ Check-in/Check-out Times")
        checkin_start = st.time_input("Check-in Start Time", value=time(8, 0))  # FIXED
        checkin_end = st.time_input("Check-in End Time", value=time(10, 0))     # FIXED
        checkout_start = st.time_input("Check-out Start Time", value=time(16, 0))  # FIXED
        checkout_end = st.time_input("Check-out End Time", value=time(18, 0))      # FIXED
        
        late_threshold = st.number_input("Late Threshold (minutes after start)", 
                                        min_value=0, max_value=120, value=15)
        early_threshold = st.number_input("Early Threshold (minutes before end)", 
                                         min_value=0, max_value=120, value=15)
    
    with col2:
        st.markdown("#### 📅 Work Schedule")
        work_days = st.multiselect("Work Days", 
                                  ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                                  default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        
        work_hours = st.slider("Daily Work Hours", min_value=4, max_value=12, value=8, step=1)
        
        st.markdown("#### 📍 Location Settings")
        require_location = st.checkbox("Require Location for Attendance", value=True)
        location_accuracy = st.slider("Minimum Location Accuracy (meters)", 
                                     min_value=10, max_value=1000, value=100, step=10)
    
    # Save button
    if st.button("💾 Save Attendance Settings", type="primary", key="save_attendance"):
        st.success("Attendance settings saved successfully!" \
        "Pending Actual Implementation")

with tab3:
    st.markdown("### Security Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔐 Authentication")
        session_timeout = st.number_input("Session Timeout (minutes)", 
                                         min_value=15, max_value=480, value=60)
        max_login_attempts = st.number_input("Max Login Attempts", 
                                            min_value=1, max_value=10, value=3)
        password_min_length = st.number_input("Minimum Password Length", 
                                            min_value=6, max_value=20, value=8)
        
        require_2fa = st.checkbox("Require Two-Factor Authentication", value=False)
        require_email_verification = st.checkbox("Require Email Verification", value=False)
    
    with col2:
        st.markdown("#### 🛡️ Data Protection")
        auto_backup = st.checkbox("Enable Automatic Backups", value=True)
        backup_frequency = st.selectbox("Backup Frequency", 
                                       ["Daily", "Weekly", "Monthly"], index=0)
        
        retention_days = st.number_input("Data Retention (days)", 
                                        min_value=30, max_value=3650, value=365)
        
        enable_audit_log = st.checkbox("Enable Audit Logging", value=True)
        log_retention = st.number_input("Log Retention (days)", 
                                       min_value=30, max_value=365, value=90)
    
    # Password policy
    st.markdown("#### 📋 Password Policy")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        require_uppercase = st.checkbox("Require Uppercase", value=True)
        require_lowercase = st.checkbox("Require Lowercase", value=True)
    with col_b:
        require_numbers = st.checkbox("Require Numbers", value=True)
        require_special = st.checkbox("Require Special Characters", value=False)
    with col_c:
        password_expiry = st.number_input("Password Expiry (days)", 
                                         min_value=0, max_value=365, value=90,
                                         help="0 = never expire")
    
    # Save button
    if st.button("💾 Save Security Settings", type="primary", key="save_security"):
        st.success("Security settings saved successfully! Pending Actual Implementation")

with tab4:
    st.markdown("### Analytics & Reporting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Report Settings")
        default_report_period = st.selectbox("Default Report Period", 
                                           ["Today", "This Week", "This Month", "This Quarter", "This Year"], 
                                           index=2)
        
        auto_generate_reports = st.checkbox("Auto-generate Monthly Reports", value=True)
        report_recipients = st.text_area("Report Recipients (emails)", 
                                        placeholder="Enter comma-separated emails",
                                        help="Who receives automated reports")
        
        st.markdown("#### 📈 Dashboard Widgets")
        show_realtime_stats = st.checkbox("Show Real-time Stats", value=True)
        show_attendance_map = st.checkbox("Show Attendance Map", value=False)
        show_employee_leaderboard = st.checkbox("Show Employee Leaderboard", value=True)
    
    with col2:
        st.markdown("#### 📧 Notification Settings")
        email_notifications = st.checkbox("Enable Email Notifications", value=True)
        notify_on_late = st.checkbox("Notify on Late Arrival", value=False)
        notify_on_absent = st.checkbox("Notify on Absence", value=True)
        notify_on_anomaly = st.checkbox("Notify on Anomaly", value=True)
        
        notification_time = st.time_input("Daily Notification Time", value=time(9, 0))  # FIXED
        
        st.markdown("#### 🎨 Display Settings")
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"], index=0)
        records_per_page = st.number_input("Records Per Page", min_value=10, max_value=100, value=25)
    
    # Save button
    if st.button("💾 Save Analytics Settings", type="primary", key="save_analytics"):
        st.success("Analytics settings saved successfully! Pending Actual Implementation")

with tab5:
    st.markdown("### 🛠️ System Maintenance")

    with st.expander("Database Maintenance", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔄 Database Operations")
            if st.button("Optimize Database", use_container_width=True, key="optimize_db"):
                conn = get_db_connection()
                conn.execute("VACUUM")
                conn.close()
                st.success("Database optimized successfully!")

            if st.button("Clear Old Logs", use_container_width=True, key="clear_logs"):
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM login_logs WHERE login_time < datetime('now', '-90 days')")
                conn.commit()
                conn.close()
                st.success("Old logs cleared successfully!")

        with col2:
            st.markdown("#### 📤 Export/Import")
            if st.button("Export All Data", use_container_width=True, key="export_data"):
                st.info("Export functionality coming soon")

            if st.button("Import Data", use_container_width=True, key="import_data"):
                st.info("Import functionality coming soon")

    with st.expander("#### Clear Records", expanded=False):
        st.warning("Deletes selected records. Use with caution.")
        table_options = ["attendance", "anomaly_log", "login_logs", "users", "employees"]
        selected_tables = st.multiselect(
            "Select tables to clear",
            options=table_options,
            default=["attendance", "anomaly_log", "login_logs"],
        )
        confirm_clear = st.checkbox("I understand this will delete records permanently", key="confirm_clear_records")
        if st.button("🗑️ Clear Selected Tables", type="secondary", use_container_width=True):
            if not selected_tables:
                st.info("Select at least one table.")
            elif not confirm_clear:
                st.warning("Please confirm before clearing records.")
            else:
                conn = get_db_connection()
                cursor = conn.cursor()
                for table in selected_tables:
                    cursor.execute(f"DELETE FROM {table}")
                conn.commit()
                conn.close()
                st.success("Selected tables cleared successfully.")

    with st.expander("#### System Information", expanded=False):
        import platform
        import sys

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🖥️ System Details")
            st.write(f"**Python Version:** {sys.version}")
            st.write(f"**Streamlit Version:** {st.__version__}")
            st.write(f"**Platform:** {platform.platform()}")

        with col2:
            st.markdown("#### 📊 Database Info")
            conn = get_db_connection()
            cursor = conn.cursor()

            tables = ['employees', 'attendance', 'anomaly_log', 'users', 'login_logs']
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    st.write(f"**{table}:** {count:,} records")
                except:
                    st.write(f"**{table}:** Table not found")

            conn.close()

with tab6:
    st.markdown("### ⚠️ Danger Zone")

    with st.expander("Advanced Operations", expanded=False):
        st.warning("⚠️ These operations are irreversible. Use with extreme caution!")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Reset All Settings", type="secondary", use_container_width=True, key="reset_settings"):
                if st.checkbox("I understand this will reset all system settings", key="confirm_reset") == True:
                    # Reset settings logic here
                    st.success("All settings have been reset to default. Pending Actual Implementation")


        with col2:
            st.info("Use System Maintenance to clear selected tables.")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings Navigation")
    
    if st.button("🔄 Refresh Settings", use_container_width=True, key="refresh_settings"):
        st.rerun()
    
    st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
    
    st.markdown("### ℹ️ Quick Help")
    
    with st.expander("Settings Guide"):
        st.markdown("""
        **Settings Categories:**
        
        - **General:** System name, timezone, formats
        - **Attendance:** Work hours, thresholds, location
        - **Security:** Authentication, passwords, backups
        - **Analytics:** Reports, notifications, display
        
        **Best Practices:**
        1. Test changes in staging first
        2. Document all changes
        3. Backup before major changes
        4. Notify users of significant changes
        """)
    
    # Save All button
    st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
    if st.button("💾 Save All Changes", type="primary", use_container_width=True, key="save_all"):
        st.success("All settings saved successfully! Pending Actual Implementation")

# Footer
st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>💼 <strong>AI-Powered Attendance System</strong></p>
        <p style="font-size: 0.9em;">System Settings Module</p>
        <p style="font-size: 0.8em;">Configurations • Model Training • Data Privacy</p>
        <p style="font-size: 0.8em;">Version 2.1 | Developed by: Itoro Udonyah (NOU234244897) | <a href="https://github.com/itoroudonyah" target="_blank">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)
