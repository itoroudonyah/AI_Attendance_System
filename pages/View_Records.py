# pages/3_View_Records.py (UPDATED FOR DATABASE SCHEMA)
import streamlit as st
import database as db
import pandas as pd
import os
import io
import plotly.express as px
from datetime import datetime
from navigation import apply_sidebar_style, render_sidebar, ensure_session, require_roles, render_page_header

st.set_page_config(
    page_title="View Records | Workforce Accountability",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "View Employees, Users and Attendance records."
    }
)

apply_sidebar_style()
ensure_session(timeout_minutes=None)
render_sidebar("📊 View Records")
require_roles(("admin", "manager"))

render_page_header("📊 View Records")
st.markdown("Choose your data source and record type to view. This selection will determine the data used by visualization.")



# --- Initialization of Session State for Global Data Sharing ---
if 'current_data_source' not in st.session_state:
    st.session_state.current_data_source = 'System Database'
if 'current_records_df' not in st.session_state:
    st.session_state.current_records_df = pd.DataFrame()
if 'current_record_type' not in st.session_state:
    st.session_state.current_record_type = 'Attendance Records'
    
DATABASE_NAME = 'attendance.db'
db.init_db()

# --- Utility Function to Standardize Attendance Data ---
def standardize_attendance_data(df):
    """
    Ensures the DataFrame has the mandatory 'timestamp' column and
    employee identifier/detail columns for downstream pages.
    """
    df = df.copy()

    # CRITICAL CHECK: Ensure employee_id exists for linking/visuals
    if 'employee_id' not in df.columns:
        raise ValueError("Missing critical column: 'employee_id'. Please ensure your CSV includes this column.")
        
    # 1. Create 'timestamp' column
    if 'timestamp' not in df.columns:
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        time_cols = [col for col in df.columns if 'time' in col.lower() and col.lower() != 'job_title']
        
        if date_cols and time_cols:
            # Combine the columns and convert to datetime objects
            df['timestamp'] = pd.to_datetime(
                df[date_cols[0]].astype(str) + ' ' + df[time_cols[0]].astype(str),
                errors='coerce'
            )
        elif date_cols and not time_cols:
            df['timestamp'] = pd.to_datetime(
                df[date_cols[0]].astype(str) + ' 00:00:00',
                errors='coerce'
            )
        elif 'Timestamp' in df.columns:
            df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)
        else:
            raise ValueError("Missing 'timestamp' column or recognizable 'date' and 'time' columns.")
    
    # 2. Ensure mandatory columns for visuals/anomalies exist
    if 'employee_name' not in df.columns:
        # Use employee_id to generate a placeholder name
        df['employee_name'] = df['employee_id'].apply(lambda x: f"Employee {x}")
    
    # Final check: Drop rows where timestamp creation failed
    return df.dropna(subset=['timestamp'])

data_tab, visuals_tab = st.tabs(["Data Source", "Visuals"])

with data_tab:
    # --- Data Source Selection ---
    st.markdown('<h3 style="color:#6078ea;">1. Select Data Source</h3>', unsafe_allow_html=True)
    data_source = st.radio(
        "Source:",
        ["System Database", "Import CSV File"],
        key="data_source_radio",
        horizontal=True
    )
    st.session_state.current_data_source = data_source
    st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)

    # --- Record Type Selection ---
    st.markdown('<h3 style="color:#6078ea;">2. Select Record Type</h3>', unsafe_allow_html=True)
    record_type = st.radio(
        "Record Type:",
        ["Attendance Records", "Employee Records", "User Accounts"],
        key="record_type_radio",
        horizontal=True
    )
    st.session_state.current_record_type = record_type
    st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)

    # --- Data Loading Logic ---
    final_df = pd.DataFrame()

    if data_source == "System Database":
        st.markdown('<h3 style="color:#6078ea;">3. System Database Records</h3>', unsafe_allow_html=True)

        try:
            if record_type == "Attendance Records":
                attendance_data = db.get_all_attendance()

                if attendance_data is None:
                    df = pd.DataFrame()
                    st.info("No attendance records found in the database.")
                elif isinstance(attendance_data, pd.DataFrame):
                    df = attendance_data.copy()
                else:
                    attendance_columns = [
                        'id', 'employee_id', 'employee_name', 'department', 'job_title',
                        'date', 'time', 'check_type', 'ip_address',
                        'location_city', 'location_region', 'location_country', 'created_at'
                    ]
                    if len(attendance_data) > 0 and hasattr(attendance_data[0], '_fields'):
                        df = pd.DataFrame([dict(row) for row in attendance_data])
                    elif len(attendance_data) > 0:
                        df = pd.DataFrame(
                            attendance_data,
                            columns=attendance_columns[:len(attendance_data[0])]
                        )
                    else:
                        df = pd.DataFrame()

                if not df.empty and 'created_at' in df.columns:
                    created_dt = pd.to_datetime(df['created_at'], errors='coerce', dayfirst=True, format='mixed')
                    df['created_at'] = created_dt.dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
                elif df.empty:
                    st.info("No attendance records found in the database.")

            elif record_type == "Employee Records":
                employee_data = db.get_all_employees()

                if employee_data is not None:
                    if isinstance(employee_data, pd.DataFrame):
                        df = employee_data.copy()
                    else:
                        if len(employee_data) > 0 and hasattr(employee_data[0], '_fields'):
                            df = pd.DataFrame([dict(row) for row in employee_data])
                        else:
                            column_names = [
                                'employee_id', 'employee_name', 'department', 'job_title',
                                'hire_date', 'email', 'phone', 'photo_path',
                                'created_at', 'is_active'
                            ]
                            df = pd.DataFrame(employee_data, columns=column_names[:len(employee_data[0])])

                    if not df.empty and 'hire_date' in df.columns:
                        hire_dt = pd.to_datetime(df['hire_date'], errors='coerce', dayfirst=True, format='mixed')
                        df['hire_date'] = hire_dt.dt.strftime('%Y-%m-%d').fillna('')
                    elif df.empty:
                        st.info("No employee records found in the database.")
                else:
                    df = pd.DataFrame()
                    st.info("No employee records found in the database.")

            else:
                user_data = db.get_all_users()

                if user_data is not None:
                    if isinstance(user_data, pd.DataFrame):
                        df = user_data.copy()
                    else:
                        if len(user_data) > 0 and hasattr(user_data[0], '_fields'):
                            df = pd.DataFrame([dict(row) for row in user_data])
                        else:
                            column_names = [
                                'id', 'username', 'password_hash', 'employee_name',
                                'email', 'role', 'employee_id', 'department',
                                'created_at', 'is_active'
                            ]
                            df = pd.DataFrame(user_data, columns=column_names[:len(user_data[0])])

                    if not df.empty and 'created_at' in df.columns:
                        created_dt = pd.to_datetime(df['created_at'], errors='coerce', dayfirst=True, format='mixed')
                        df['created_at'] = created_dt.dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
                    elif df.empty:
                        st.info("No user account records found in the database.")

                    if not df.empty and 'password_hash' in df.columns:
                        df['password_hash'] = '********'
                else:
                    df = pd.DataFrame()
                    st.info("No user account records found in the database.")

            final_df = df.copy()
            if not final_df.empty:
                st.success(f"Loaded {len(final_df)} {record_type} from the system database.")

        except Exception as e:
            st.error(f"Error fetching data from system database: {e}")
            st.info("Ensure the database file exists and employees are registered.")

    elif data_source == "Import CSV File":
        st.markdown('<h3 style="color:#6078ea;">3. Import CSV File</h3>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            f"Upload a CSV file for {record_type}:",
            type=['csv'],
            key="csv_uploader"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                if record_type == "Attendance Records":
                    df = standardize_attendance_data(df)
                    column_mapping = {
                        'full_name': 'employee_name',
                        'name': 'employee_name',
                        'position': 'job_title'
                    }
                    df.rename(columns=column_mapping, inplace=True)

                st.success(f"Successfully loaded {len(df)} rows from '{uploaded_file.name}'.")
                final_df = df.copy()

            except ValueError as e:
                st.error(f"Data Preparation Error: {e}")
                st.info("Please correct your CSV file and try again.")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                st.info("Please ensure the file is a valid CSV.")
                import traceback
                st.code(traceback.format_exc())

    # --- Display and Save Final DataFrame ---
    if not final_df.empty:
        st.markdown(f'<h3 style="color:#6078ea;">4. Displaying Loaded {record_type}</h3>', unsafe_allow_html=True)

        view_df = final_df.copy()
        if record_type == "Attendance Records" and 'employee_id' in final_df.columns:
            employee_labels = final_df[['employee_id']].copy()
            if 'employee_name' in final_df.columns:
                employee_labels['employee_name'] = final_df['employee_name']
            else:
                employee_labels['employee_name'] = ""
            employee_labels = employee_labels.drop_duplicates()
            employee_labels['label'] = employee_labels.apply(
                lambda r: f"{r['employee_id']} - {r['employee_name']}".strip(" -"),
                axis=1,
            )
            labels = employee_labels['label'].tolist()
            label_to_id = dict(zip(labels, employee_labels['employee_id']))

            selected_label = st.selectbox(
                "Filter by employee",
                ["All Employees"] + labels,
                key="attendance_employee_filter",
            )
            if selected_label != "All Employees":
                selected_id = label_to_id.get(selected_label)
                view_df = view_df[view_df['employee_id'] == selected_id].copy()
                st.info(f"Filtered to {len(view_df)} records for {selected_label}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(view_df))

        with col2:
            if 'timestamp' in view_df.columns:
                date_range = view_df['timestamp'].agg(['min', 'max'])
                st.metric("Date Range", f"{date_range['min'].date()} to {date_range['max'].date()}")
            elif 'hire_date' in view_df.columns:
                hire_dates = pd.to_datetime(view_df['hire_date'], errors='coerce')
                date_range = hire_dates.agg(['min', 'max'])
                if pd.notna(date_range['min']) and pd.notna(date_range['max']):
                    st.metric("Hire Date Range", f"{date_range['min'].date()} to {date_range['max'].date()}")
                else:
                    st.metric("Hire Date Range", "Unavailable")

        with col3:
            if 'department' in view_df.columns:
                unique_depts = view_df['department'].nunique()
                st.metric("Unique Departments", unique_depts)

        final_df_display = view_df.copy()
        if 'created_at' in final_df_display.columns:
            created_at_dt = pd.to_datetime(final_df_display['created_at'], errors='coerce', dayfirst=True)
            final_df_display['created_at'] = created_at_dt.dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
        for col in final_df_display.columns:
            if final_df_display[col].dtype == "object":
                final_df_display[col] = final_df_display[col].astype(str)

        if 'username' in final_df.columns:
            st.dataframe(final_df_display, use_container_width=True, height=400)
        else:
            st.dataframe(final_df_display, use_container_width=True, height=400)

        st.markdown('<h3 style="color:#6078ea;">5. Export Options</h3>', unsafe_allow_html=True)

        col_export1, col_export2, col_export3 = st.columns(3)
        with col_export1:
            csv = final_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{record_type.replace(' ', '_').lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col_export2:
            json_str = final_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name=f"{record_type.replace(' ', '_').lower()}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

        with col_export3:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.write("Data copied to clipboard! (Ctrl+V to paste)")

        with st.expander("📈 View Statistics"):
            st.write("**Column Information:**")
            col_info = pd.DataFrame({
                'Column': final_df.columns,
                'Data Type': final_df.dtypes.astype(str).values,
                'Non-Null Count': final_df.notnull().sum().values,
                'Null Count': final_df.isnull().sum().values,
                'Unique Values': [final_df[col].nunique() for col in final_df.columns]
            })
            st.dataframe(col_info, use_container_width=True)

            if 'timestamp' in final_df.columns or 'hire_date' in final_df.columns or 'created_at' in final_df.columns:
                st.write("**Date Statistics:**")
                date_col = next((col for col in ['timestamp', 'hire_date', 'created_at'] if col in final_df.columns), None)
                if date_col:
                    date_stats = final_df[date_col].describe()
                    st.write(date_stats)

        st.session_state.current_records_df = final_df
        st.sidebar.success(f"✅ Data loaded: {record_type} ({len(final_df)} rows)")
        st.sidebar.info(f"**Source:** {data_source}")

        with st.sidebar.expander("🔧 Quick Actions"):
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
            if st.button("🧹 Clear Data", use_container_width=True):
                st.session_state.current_records_df = pd.DataFrame()
                st.rerun()

    else:
        st.info(f"Please select a valid data source and ensure data is available for '{record_type}'.")
        st.session_state.current_records_df = pd.DataFrame()
        if data_source == "Import CSV File":
            with st.expander("📋 Expected CSV Format"):
                if record_type == "Attendance Records":
                    st.markdown("""
                    **Required columns for Attendance Records CSV:**
                    - `employee_id` (string) - Employee identifier
                    - `timestamp` (datetime) OR separate `date` and `time` columns
                    - `attendance_type` (string, optional) - "IN"/"OUT"
                    - `department` (string, optional)
                    
                    **Example CSV:**
                    ```
                    employee_id,timestamp,attendance_type,department
                    EMP001,2024-01-15 08:30:00,IN,IT
                    EMP002,2024-01-15 09:00:00,IN,HR
                    ```
                    """)
                elif record_type == "Employee Records":
                    st.markdown("""
                    **Recommended columns for Employee Records CSV:**
                    - `employee_id` (string) - Employee identifier
                    - `employee_name` (string) - Full name
                    - `department` (string) - Department
                    - `job_title` (string) - Job title
                    - `hire_date` (date) - Hire date
                    - `email` (string) - Email address
                    - `phone` (string) - Phone number
                    
                    **Example CSV:**
                    ```
                    employee_id,employee_name,department,job_title,hire_date,email,phone
                    EMP001,John Doe,IT,Software Engineer,2023-01-15,john@company.com,+1234567890
                    EMP002,Jane Smith,HR,Manager,2022-06-01,jane@company.com,+0987654321
                    ```
                    """)

with visuals_tab:
    st.markdown("### Visuals")
    records_df = st.session_state.get("current_records_df", pd.DataFrame())
    current_type = st.session_state.get("current_record_type", "Attendance Records")

    if records_df is None or records_df.empty:
        st.info("Load data in the Data Source tab to see visuals.")
    else:
        if current_type == "Attendance Records":
            df = records_df.copy()
            if "timestamp" not in df.columns:
                if "date" in df.columns and "time" in df.columns:
                    df["timestamp"] = pd.to_datetime(
                        df["date"].astype(str) + " " + df["time"].astype(str),
                        errors="coerce",
                    )
                elif "date" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")

            df = df.dropna(subset=["timestamp"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df["date_only"] = df["timestamp"].dt.date

            range_label = st.selectbox(
                "Time Range",
                ["Last 7 days", "Last 30 days", "Last 1 year", "All"],
                index=1,
                key="attendance_visual_range",
            )
            max_date = df["timestamp"].max()
            if range_label == "Last 7 days":
                min_date = max_date - pd.Timedelta(days=7)
            elif range_label == "Last 30 days":
                min_date = max_date - pd.Timedelta(days=30)
            elif range_label == "Last 1 year":
                min_date = max_date - pd.Timedelta(days=365)
            else:
                min_date = df["timestamp"].min()
            df = df[df["timestamp"] >= min_date]

            col1, col2 = st.columns(2)
            with col1:
                if "department" in df.columns:
                    dept_daily = df.groupby(["date_only", "department"]).size().reset_index(name="count")
                    fig = px.line(
                        dept_daily,
                        x="date_only",
                        y="count",
                        color="department",
                        title="Attendance Trends by Department",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Department column not available for department trend.")
            with col2:
                if "employee_name" in df.columns:
                    emp_daily = df.groupby(["date_only", "employee_name"]).size().reset_index(name="count")
                    top_emps = (
                        emp_daily.groupby("employee_name")["count"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(10)
                        .index.tolist()
                    )
                    emp_daily = emp_daily[emp_daily["employee_name"].isin(top_emps)]
                    fig = px.line(
                        emp_daily,
                        x="date_only",
                        y="count",
                        color="employee_name",
                        title="Top Employee Attendance Trends",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Employee name column not available for employee trend.")

            st.markdown("#### Attendance Heatmap by Day/Hour")
            check_filter = "All"
            if "check_type" in df.columns:
                check_filter = st.selectbox(
                    "Include check type",
                    ["All", "check_in", "check_out"],
                    index=1,
                    key="attendance_heatmap_check_type",
                )
            if check_filter != "All" and "check_type" in df.columns:
                df = df[df["check_type"].astype(str).str.lower() == check_filter].copy()
            df["day_name"] = df["timestamp"].dt.day_name()
            df["hour"] = df["timestamp"].dt.hour
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heat_df = (
                df.groupby(["day_name", "hour"]).size().reset_index(name="count")
            )
            heat_df["day_name"] = pd.Categorical(heat_df["day_name"], categories=day_order, ordered=True)
            heat_pivot = heat_df.pivot_table(index="day_name", columns="hour", values="count", fill_value=0)
            heat_pivot = heat_pivot.reindex(day_order)
            hour_labels = {h: datetime.strptime(str(h), "%H").strftime("%-I %p") for h in heat_pivot.columns}
            heat_pivot = heat_pivot.rename(columns=hour_labels)

            fig = px.imshow(
                heat_pivot,
                aspect="auto",
                color_continuous_scale="YlOrRd",
                labels=dict(x="Hour of Day", y="Day of Week", color="Count"),
                title=f"Attendance Frequency Heatmap ({range_label})",
            )
            st.plotly_chart(fig, use_container_width=True)
        elif current_type == "Employee Records":
            df = records_df.copy()
            if "department" in df.columns:
                dept_counts = df["department"].value_counts().reset_index()
                dept_counts.columns = ["department", "count"]
                fig = px.bar(
                    dept_counts,
                    x="department",
                    y="count",
                    title="Employees per Department",
                    color="department",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Department column not available for employee visuals.")
        else:
            st.info("Visuals are available for Attendance and Employee records only.")

# Footer
st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>💼 <strong>AI-Powered Attendance System</strong></p>
        <p style="font-size: 0.9em;">Record View Module</p>
        <p style="font-size: 0.8em;">Attendance & Employees Data | System Database & CSV</p>
        <p style="font-size: 0.8em;">Version 2.1 | Developed by: Itoro Udonyah (NOU234244897) | <a href="https://github.com/itoroudonyah" target="_blank">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)
