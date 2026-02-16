# My_Attendance.py - Personal attendance history for regular users
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from navigation import apply_sidebar_style, render_sidebar, ensure_session, require_roles, render_page_header

st.set_page_config(
    page_title="My Attendance History",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_sidebar_style()
ensure_session(timeout_minutes=None)
render_sidebar("👤 My Attendance")
require_roles(("admin", "manager", "user", "employee"))

render_page_header("👤 My Attendance")

# Database connection
def get_db_connection():
    import database as db
    return db.get_connection()

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #4facfe;
    }
    .attendance-record {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    .late-record {
        border-left-color: #ff6b6b;
    }
    .early-record {
        border-left-color: #ffd93d;
    }
    .filter-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Page Header
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader(f"Welcome, {st.session_state.employee_name}")
    st.text("Review and analyze your personal attendance records over time.")

    if st.session_state.employee_id:
        st.caption(f"Employee ID: {st.session_state.employee_id}")
    if st.session_state.department:
        st.caption(f"Department: {st.session_state.department}")

st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)

# Function to get user's attendance
def get_user_attendance(employee_id, start_date=None, end_date=None):
    conn = get_db_connection()
    
    query = """
        SELECT 
            a.id,
            a.employee_id,
            a.date,
            a.time,
            a.check_type,
            a.latitude,
            a.longitude,
            COALESCE(e.employee_name, a.employee_name) AS employee_name,
            e.job_title
        FROM attendance a
        LEFT JOIN employees e ON a.employee_id = e.employee_id
        WHERE a.employee_id = ?
    """
    
    params = [employee_id]
    query += " ORDER BY a.date DESC, a.time DESC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if not df.empty and 'date' in df.columns and 'time' in df.columns:
        # Support both ISO (YYYY-MM-DD) and day-first (DD/MM/YYYY) date formats.
        date_str = df['date'].astype(str).str.strip()
        time_str = df['time'].astype(str).str.strip()
        iso_mask = date_str.str.match(r"^\d{4}-\d{2}-\d{2}$")

        parsed_dates = pd.Series(pd.NaT, index=df.index)
        if iso_mask.any():
            parsed_dates.loc[iso_mask] = pd.to_datetime(date_str[iso_mask], format="%Y-%m-%d", errors="coerce")
        if (~iso_mask).any():
            parsed_dates.loc[~iso_mask] = pd.to_datetime(date_str[~iso_mask], format="%d/%m/%Y", errors="coerce")

        df['timestamp'] = pd.to_datetime(
            parsed_dates.dt.strftime("%Y-%m-%d") + " " + time_str,
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['hour'] = df['timestamp'].dt.hour
        df['date_only'] = df['timestamp'].dt.date

        # Apply date range filter in pandas to handle mixed date formats.
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date).date()
            end_dt = pd.to_datetime(end_date).date()
            df = df[(df['date_only'] >= start_dt) & (df['date_only'] <= end_dt)]
        
        # Determine if check-in is late (after 9 AM) or early (before 8 AM)
        df['status'] = 'On Time'
        df.loc[(df['check_type'] == 'check_in') & (df['hour'] >= 9), 'status'] = 'Late'
        df.loc[(df['check_type'] == 'check_in') & (df['hour'] < 8), 'status'] = 'Early'
    
    return df

# Function to calculate attendance stats
def calculate_stats(df):
    stats = {
        'total_days': 0,
        'present_days': 0,
        'late_days': 0,
        'early_days': 0,
        'avg_checkin_time': 'N/A',
        'attendance_rate': 0
    }
    
    if not df.empty:
        # Get check-ins only
        checkins = df[df['check_type'] == 'check_in']
        
        if not checkins.empty:
            stats['total_days'] = checkins['date_only'].nunique()
            stats['present_days'] = stats['total_days']
            
            # Count late and early check-ins
            stats['late_days'] = (checkins['status'] == 'Late').sum()
            stats['early_days'] = (checkins['status'] == 'Early').sum()
            
            # Calculate average check-in time
            avg_hour = checkins['hour'].mean()
            if pd.notna(avg_hour):
                hour = int(avg_hour)
                minute = int((avg_hour - hour) * 60)
                stats['avg_checkin_time'] = f"{hour:02d}:{minute:02d}"
            
            # Calculate attendance rate (this month)
            today = datetime.now()
            days_in_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            days_in_month = days_in_month.day
            stats['attendance_rate'] = (stats['present_days'] / days_in_month * 100) if days_in_month > 0 else 0
    
    return stats

# Main content
employee_id = st.session_state.employee_id
if employee_id is not None:
    employee_id = str(employee_id).strip()

if not employee_id:
    st.warning("⚠️ Your account is not linked to an employee record. Please contact your administrator.")
    
    # Show user profile information
    with st.expander("📋 My Profile Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Username:** {st.session_state.username}")
            st.info(f"**Full Name:** {st.session_state.employee_name}")
        with col2:
            if st.session_state.department:
                st.info(f"**Department:** {st.session_state.department}")
            if 'email' in st.session_state and st.session_state.email:
                st.info(f"**Email:** {st.session_state.email}")
    
    st.stop()

# Date range filter
st.markdown("### 📅 Filter Attendance Records")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    # Default to last 30 days
    default_end = datetime.now()
    default_start = default_end - timedelta(days=30)
    
    start_date = st.date_input("Start Date", value=default_start)
    
with col2:
    end_date = st.date_input("End Date", value=default_end)
    
with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    apply_filter = st.button("Apply Filter", use_container_width=True)

# Persist applied filters so the button actually controls the query
if "attendance_filter_start" not in st.session_state:
    st.session_state.attendance_filter_start = start_date
if "attendance_filter_end" not in st.session_state:
    st.session_state.attendance_filter_end = end_date

if apply_filter:
    st.session_state.attendance_filter_start = start_date
    st.session_state.attendance_filter_end = end_date

# Use applied filters for queries
applied_start = st.session_state.attendance_filter_start
applied_end = st.session_state.attendance_filter_end

# Guard against inverted ranges
if applied_end < applied_start:
    st.warning("End Date is earlier than Start Date. Using Start Date as End Date.")
    applied_end = applied_start

# Get attendance data
attendance_df = get_user_attendance(
    employee_id, 
    applied_start.strftime("%Y-%m-%d"), 
    applied_end.strftime("%Y-%m-%d")
)

# Calculate statistics
stats = calculate_stats(attendance_df)

# Display metrics
st.markdown("### 📊 Attendance Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin: 0; color: #4facfe;">{stats['present_days']}</h3>
        <p style="margin: 0.5rem 0 0 0; color: #666; font-weight: bold;">Days Present</p>
        <p style="margin: 0; font-size: 0.9em; color: #666;">Out of {stats['total_days']} days</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin: 0; color: {'#ff6b6b' if stats['late_days'] > 0 else '#28a745'};">{stats['late_days']}</h3>
        <p style="margin: 0.5rem 0 0 0; color: #666; font-weight: bold;">Late Days</p>
        <p style="margin: 0; font-size: 0.9em; color: #666;">This period</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin: 0; color: #ffd93d;">{stats['early_days']}</h3>
        <p style="margin: 0.5rem 0 0 0; color: #666; font-weight: bold;">Early Days</p>
        <p style="margin: 0; font-size: 0.9em; color: #666;">This period</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin: 0; color: #764ba2;">{stats['avg_checkin_time']}</h3>
        <p style="margin: 0.5rem 0 0 0; color: #666; font-weight: bold;">Avg. Check-in</p>
        <p style="margin: 0; font-size: 0.9em; color: #666;">Time</p>
    </div>
    """, unsafe_allow_html=True)

# Visualization section
st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
st.markdown("### 📈 Attendance Visualization")

if not attendance_df.empty:
    tab1, tab2, tab3 = st.tabs(["📅 Daily Records", "📊 Statistics", "🕒 Time Analysis"])
    
    with tab1:
        # Show attendance records
        st.markdown("#### Your Attendance Records")
        
        daily_records = attendance_df.sort_values('timestamp', ascending=False).copy()
        daily_records['Date'] = daily_records['timestamp'].dt.strftime("%A, %B %d, %Y")
        daily_records['Time'] = daily_records['timestamp'].dt.strftime("%I:%M %p")
        daily_records['Type'] = daily_records['check_type'].map(
            {'check_in': 'Check-in', 'check_out': 'Check-out'}
        ).fillna(daily_records['check_type'])
        daily_records['Status'] = daily_records['status'].fillna('On Time')
        daily_records['Day'] = daily_records['day_of_week']

        table_cols = ['Date', 'Time', 'Type', 'Status', 'Day']
        st.dataframe(
            daily_records[table_cols],
            use_container_width=True,
            hide_index=True,
        )
    
    with tab2:
        # Attendance statistics chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for check-in status
            if not attendance_df[attendance_df['check_type'] == 'check_in'].empty:
                status_counts = attendance_df[attendance_df['check_type'] == 'check_in']['status'].value_counts()
                fig1 = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Check-in Status Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig1, use_container_width=True)
                st.caption("Share of your check-ins that were on time, early, or late.")
        
        with col2:
            # Bar chart for weekly pattern
            if not attendance_df.empty:
                weekly_counts = attendance_df['day_of_week'].value_counts()
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekly_counts = weekly_counts.reindex(days_order, fill_value=0)
                
                fig2 = px.bar(
                    x=weekly_counts.index,
                    y=weekly_counts.values,
                    title="Attendance by Day of Week",
                    labels={'x': 'Day', 'y': 'Count'},
                    color=weekly_counts.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.caption("Number of check-ins by day of week to show your attendance pattern.")
    
    with tab3:
        # Time analysis
        st.markdown("#### Check-in Time Distribution")
        
        if not attendance_df[attendance_df['check_type'] == 'check_in'].empty:
            checkins = attendance_df[attendance_df['check_type'] == 'check_in']
            
            # Histogram of check-in times
            fig3 = px.histogram(
                checkins,
                x='hour',
                nbins=24,
                title="Check-in Time Distribution (24-hour format)",
                labels={'hour': 'Hour of Day', 'count': 'Number of Check-ins'},
                color_discrete_sequence=['#4facfe']
            )
            fig3.add_vline(x=9, line_dash="dash", line_color="red", annotation_text="9 AM")
            fig3.add_vline(x=8, line_dash="dash", line_color="green", annotation_text="8 AM")
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("Distribution of your check-in times across the day.")
            
            # Monthly trend
            if len(checkins) > 7:
                monthly_trend = checkins.resample('D', on='timestamp').size().reset_index()
                monthly_trend.columns = ['date', 'count']
                
                fig4 = px.line(
                    monthly_trend,
                    x='date',
                    y='count',
                    title="Attendance Trend Over Time",
                    labels={'date': 'Date', 'count': 'Daily Check-ins'},
                    line_shape='spline'
                )
                st.plotly_chart(fig4, use_container_width=True)
                st.caption("Daily check-in trend over the selected period.")
    
    # Export option
    st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("📥 Export Data", use_container_width=True):
            csv = attendance_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"attendance_{employee_id}_{start_date}_{end_date}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.info("📭 No attendance records found for the selected period.")
    
    # Show help for new users
    with st.expander("❓ How to get started with attendance?", expanded=True):
        st.markdown("""
        ### First Time User Guide:
        
        1. **Take Attendance**  
           Go to the "Take Attendance" page to mark your first check-in
        
        2. **Regular Check-ins**  
           Remember to check in daily when you arrive at work
        
        3. **Check-out**  
           Don't forget to check out when leaving
        
        4. **View Records**  
           Your attendance records will appear here once you start using the system
        
        **Note:** If you believe there should be records but none are showing, 
        please contact your system administrator.
        """)

# Sidebar
with st.sidebar:
    st.markdown("### ℹ️ Quick Info")
    
    st.markdown(f"""
    **Employee ID:** {employee_id}
    
    **Full Name:** {st.session_state.employee_name}
    
    **Role:** {st.session_state.user_role.title()}
    
    **Records Found:** {len(attendance_df)} entries
    """)
    
    st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)

    # Help section
    with st.expander("❓ Need Help?"):
        st.markdown("""
        **Common Questions:**
        
        - **Missing records?** Contact your administrator
        - **Wrong time?** System uses server time
        - **Can't check in?** Ensure camera permissions are granted
        
        **Contact:** Your system administrator
        """)

# Footer
st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>💼 <strong>AI-Powered Attendance System</strong></p>
        <p style="font-size: 0.9em;">Facial Recognition Attendance Module</p>
        <p style="font-size: 0.8em;">Automated Check-in/Check-out • Real-time Processing • Location Tracking</p>
                 <p style="font-size: 0.8em;">Version 2.1 | Developed by: Itoro Udonyah (NOU234244897) | <a href="https://github.com/itoroudonyah" target="_blank">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)
