# Manage_Employees.py - COMPLETE VERSION WITH IMPROVED CAMERA
import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import os
import sys
import sqlite3
import pandas as pd
from PIL import Image
import io
import tempfile
import time
import re
from database import get_connection
from navigation import apply_sidebar_style, render_sidebar, ensure_session, require_roles, render_page_header

st.set_page_config(
    page_title="Manage Employees",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_sidebar_style()
ensure_session(timeout_minutes=None)
render_sidebar("👥 Manage Employees")
require_roles(("admin",))

# Add this near the top of your file
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "📋 View Employees"

def _encode_photo_path(photo_path: str):
    """Detect a face in a saved photo and return a single encoding."""
    try:
        image = fr_module.face_recognition.load_image_file(photo_path)
        locations = fr_module.face_recognition.face_locations(image)
        if not locations:
            locations = fr_module.face_recognition.face_locations(
                image, number_of_times_to_upsample=2, model="hog"
            )
        if not locations:
            # Try enlarging small images.
            h, w = image.shape[:2]
            resized = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            locations = fr_module.face_recognition.face_locations(resized)
            image = resized if locations else image

        encodings = (
            fr_module.face_recognition.face_encodings(image, locations, num_jitters=2)
            if locations
            else []
        )
        return (encodings[0] if encodings else None), len(locations)
    except Exception:
        return None, 0

# Add this function near the top of your file (after imports)
def clear_all_caches():
    """Clear all Streamlit caches to ensure fresh data"""
    import streamlit as st
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # Also clear specific caches if you know their names
    try:
        # Clear the load_known_faces cache if it exists
        if 'load_known_faces' in st.session_state:
            del st.session_state.load_known_faces
    except:
        pass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import database as db
    import face_recognition_module as fr_module
except ImportError:
    st.error("Required modules not found. Please ensure database.py and face_recognition_module.py are in the correct directory.")
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .employee-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #764ba2;
    }
    .camera-preview {
        border: 3px solid #764ba2;
        border-radius: 10px;
        padding: 5px;
        background: #f8f9fa;
    }
    .success-card {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .photo-frame {
        border: 3px solid #764ba2;
        border-radius: 10px;
        padding: 10px;
        background: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .search-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid #4facfe;
    }
    .edit-mode-header {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        color: white;
        padding: 0.4rem 0.6rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        border: 2px solid #f5576c;
        max-width: 50%;
        margin-left: auto;
        margin-right: auto;
    }
    .edit-mode-header h2 {
        font-size: 1.2rem;
        margin: 0;
    }
    .edit-mode-header p {
        font-size: 0.8rem;
        margin: 0.3rem 0 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
db.init_db()

# Database connection
def get_db_connection():
    conn = get_connection()  # ✅ ensures row_factory
    cursor = conn.cursor()
    return conn

if "delete_success_msg" in st.session_state:
    st.success(st.session_state.delete_success_msg)
    del st.session_state.delete_success_msg

if "delete_error_msg" in st.session_state:
    st.error(st.session_state.delete_error_msg)
    del st.session_state.delete_error_msg

# Function to get departments from database
def get_departments_from_db():
    """Get unique departments from employees table"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT DISTINCT department FROM employees WHERE department IS NOT NULL AND department != '' ORDER BY department")
        departments = [row[0] for row in cursor.fetchall()]
    except Exception as e:
        print(f"Error fetching departments: {e}")
        departments = []
    
    conn.close()
    return departments

# Gamma correction function
def adjust_gamma(image, gamma=1.0):
    """Apply gamma correction to image"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Session state initialization
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'capture_count' not in st.session_state:
    st.session_state.capture_count = 0
if 'employee_data' not in st.session_state:
    st.session_state.employee_data = {}
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'edit_employee_id' not in st.session_state:
    st.session_state.edit_employee_id = None
if 'last_frame_bytes' not in st.session_state:
    st.session_state.last_frame_bytes = None
if 'camera_stop_reason' not in st.session_state:
    st.session_state.camera_stop_reason = ""
if 'frame_mean_brightness' not in st.session_state:
    st.session_state.frame_mean_brightness = 0
if 'selected_employee_id' not in st.session_state:
    st.session_state.selected_employee_id = None
# Add these to session state initialization
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'edit_employee_id' not in st.session_state:
    st.session_state.edit_employee_id = None
if 'employee_data' not in st.session_state:
    st.session_state.employee_data = {}

# Header
render_page_header("👥 Employee Management")

# Main tabs - UPDATED TO HANDLE QUERY PARAMS
# Check if we should automatically select Tab 2 for editing
query_params = st.query_params

# Set default tab
default_tab = "📋 View Employees"

# Check if we're coming from an edit action
if "tab" in query_params and query_params["tab"] == "add_edit":
    default_tab = "➕ Add New Employee"
    # Ensure edit mode is set
    if "edit_id" in query_params:
        edit_id = query_params["edit_id"]
        if not st.session_state.edit_mode:
            # Load employee data if not already loaded
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM employees WHERE employee_id = ?", (edit_id,))
            employee_full = cursor.fetchone()
            conn.close()
            
            if employee_full:
                st.session_state.edit_mode = True
                st.session_state.edit_employee_id = edit_id
                st.session_state.employee_data = {
                'employee_id': employee_full[0],
                'employee_name': employee_full[1],
                'department': employee_full[2],   # Now index 2 (was 3)
                'job_title': employee_full[3],    # Now index 3 (was 5)
                'hire_date': employee_full[4],    # Now index 4 (was 6)
                'email': employee_full[5],        # Now index 5 (was 7)
                'phone': employee_full[6],        # Now index 6 (was 8)
                'photo_path': employee_full[7],   # Now index 7 (was 9)
                'is_active': employee_full[9]     # Now index 9 (was 11)
            }

#def handle_delete_employee(emp_id, emp_name):
def handle_delete_employee(emp_id, emp_name):
    print(f"DEBUG: Attempting to delete ID: '{emp_id}'")



    success, message = db.delete_employee(emp_id)

    if success:
        st.session_state.reset_employee_ui = True
        st.session_state.selected_employee_id = None

        st.cache_data.clear()
        st.cache_resource.clear()

        st.session_state.delete_success_msg = f"✅ {message}"
    else:
        st.session_state.delete_error_msg = f"❌ {message}"

def handle_delete_employee(emp_id, emp_name):
    print(f"DEBUG: Attempting to delete ID: '{emp_id}'")

    # Check emp_id exists
    if not emp_id:
        st.session_state.delete_error_msg = "❌ No employee selected for deletion or Invalid employee ID."
        return
    
    # 1. Attempt database deletion
    success, message = db.delete_employee(emp_id)

    if success:
        # 2. Set reset flag instead of touching widget state
        st.session_state.reset_employee_ui = True
        st.session_state.selected_employee_id = None
        
        # 3. Clear caches
        st.cache_data.clear()
        st.cache_resource.clear()

        # Store success message to show after rerun
        st.session_state.delete_success_msg = f"✅ {message}"
    else:
        st.session_state.delete_error_msg = f"❌ {message}"

    print(f"DEBUG: Attempting to delete ID: '{emp_id}'") # Check your terminal/console

    
    success, message = db.delete_employee(emp_id)
        
# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 View Employees", "➕ Add New Employee", "📸 Enroll Employee Face", "🔧 Database & Photo Fix Tools"])

if query_params.get("tab") == "add_edit":
    js = """
    <script>
        setTimeout(() => {
            const tabs = window.parent.document.querySelectorAll('[data-testid="stTabs"] button');
            if (tabs.length >= 2) {
                tabs[1].click();
            }
            let attempts = 0;
            const scrollTimer = setInterval(() => {
                const target = window.parent.document.querySelector('#add-edit-form');
                if (target) {
                    target.scrollIntoView({behavior: 'smooth', block: 'start'});
                    clearInterval(scrollTimer);
                }
                attempts += 1;
                if (attempts > 10) {
                    clearInterval(scrollTimer);
                }
            }, 150);
        }, 100);
    </script>
    """
    st.components.v1.html(js, height=0)

# TAB 1: View Employees - UPDATED TO TABULAR FORMAT
with tab1:
    #st.subheader("📊 Employee Directory")
    st.markdown('<h3 style="color:#6078ea;">📊 Employee Directory</h3>', unsafe_allow_html=True)
    # Search and filter
    st.markdown('<div class="search-card">', unsafe_allow_html=True)
    col_search1, col_search2, col_search3, col_search4 = st.columns(4)
    
    with col_search1:
        user_search = st.text_input("🔍 Search employees", 
                                  placeholder="Search by name, ID or department...", 
                                  key="employee_search")
    
    with col_search2:
        # Get departments for filter
        departments = get_departments_from_db()
        if not departments:
            departments = ["All Departments"]
        else:
            departments = ["All Departments"] + departments
        
        department_filter = st.selectbox(
            "Filter by Department", 
            departments, 
            key="dept_filter"
        )
    
    with col_search3:
        status_filter = st.selectbox(
            "Filter by Status", 
            ["All", "Active", "Inactive"], 
            key="status_filter"
        )
    
    with col_search4:
        enrollment_filter = st.selectbox(
            "Filter by Enrollment", 
            ["All", "Enrolled", "Not Enrolled"], 
            key="enrollment_filter"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Get employees from database with filters
    # Get employees from database with filters - WITHOUT CACHING
    conn = get_db_connection()
    cursor = conn.cursor()

    query = '''
        SELECT employee_id, employee_name, department, job_title, 
            hire_date, email, phone, photo_path, created_at, is_active
        FROM employees WHERE 1=1
    '''
    params = []

    # Apply search filter
    if user_search:
        query += " AND (employee_name LIKE ? OR employee_id LIKE ? OR department LIKE ? OR job_title LIKE ?)"
        search_term = f"%{user_search}%"
        params.extend([search_term, search_term, search_term, search_term])

    # Apply department filter
    if department_filter != "All Departments":
        query += " AND department = ?"
        params.append(department_filter)

    # Apply status filter
    if status_filter == "Active":
        query += " AND is_active = 1"
    elif status_filter == "Inactive":
        query += " AND is_active = 0"

    query += " ORDER BY employee_name"
    cursor.execute(query, params)
    employees = cursor.fetchall()
    
    # Get user roles for display
    employee_roles = {}
    if employees:
        employee_ids = [emp[0] for emp in employees]
        placeholders = ','.join(['?' for _ in employee_ids])
        cursor.execute(f"SELECT employee_id, role FROM users WHERE employee_id IN ({placeholders})", employee_ids)
        for row in cursor.fetchall():
            employee_roles[row[0]] = row[1]
    
    if employees:
        # Convert to DataFrame for table display
        employee_data = []
        for employee in employees:
            # NEW: 10 columns after removing full_name and position
            employee_id, employee_name, department, job_title, hire_date, email, phone, photo_path, created_at, is_active = employee[:10]
            
            # Check enrollment status
            encoding_file = os.path.join(fr_module.ENCODINGS_DIR, f"{employee_id}.pkl")
            enrolled = "✅" if os.path.exists(encoding_file) else "❌"
            
            # Get user role
            role = employee_roles.get(employee_id, "Not set")
            
            # Status indicator
            status = "✅ Active" if is_active == 1 else "❌ Inactive"
            
            employee_data.append({
                "Employee ID": employee_id,
                "Name": employee_name,
                "Department": department or "N/A",
                "Job Title": job_title or "N/A",  # Changed from "Position"
                "Role": role,
                "Status": status,
                "Enrolled": enrolled,
                "Email": email or "N/A",
                "Phone": phone or "N/A",
                "Hire Date": hire_date or "N/A"
            })
        
        # Create DataFrame
        df = pd.DataFrame(employee_data)
        
        # Apply enrollment filter after DataFrame creation
        if enrollment_filter == "Enrolled":
            df = df[df["Enrolled"] == "✅"]
        elif enrollment_filter == "Not Enrolled":
            df = df[df["Enrolled"] == "❌"]
        
        # Add this right after the search filters
        col_refresh, col_export, col_space = st.columns([1, 1, 4])

        with col_refresh:
            if st.button("🔄 Refresh Data", key="refresh_data_btn", use_container_width=True):
                clear_all_caches()
                if 'selected_employee_id' in st.session_state:
                    del st.session_state.selected_employee_id
                st.rerun()

        # Export button
        # col_export, col_space = st.columns([1, 5])
        with col_export:
            if st.button("📥 Export CSV", use_container_width=True):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"employees_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Display the table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Enrolled": st.column_config.Column(
                    width="small",
                    help="Face enrollment status"
                ),
                "Status": st.column_config.Column(
                    width="small",
                    help="Employee status"
                )
            }
        )
        
        # Action buttons for selected employee
        #st.subheader("🛠️ Employee Actions")
        st.markdown('<h3 style="color:#6078ea;">🛠️ Employee Actions</h3>', unsafe_allow_html=True)

        # Create a dropdown for employee selection
        if not df.empty:
            # Create display names for dropdown
            employee_options = []
            for _, row in df.iterrows():
                # Make sure we're accessing columns correctly
                emp_id = row.get('Employee ID', 'N/A')
                emp_name = row.get('Name', 'N/A')
                emp_dept = row.get('Department', 'N/A')
                employee_options.append(f"{emp_id} - {emp_name} ({emp_dept})")
            
            # Add a "Select employee..." placeholder
            employee_options_with_placeholder = ["Select employee..."] + employee_options
            
            # ---------- RESET BLOCK (PLACE THIS FIRST) ----------
            if st.session_state.get("reset_employee_ui", False):
                st.session_state.employee_action_select = "Select employee..."
                st.session_state.reset_employee_ui = False
                
            col_select, col_space = st.columns([3, 1])
            with col_select:
                selected_display = st.selectbox(
                    "Select Employee for Actions:",
                    options=employee_options_with_placeholder,
                    key="employee_action_select"
                )
            
            if selected_display != "Select employee...":
                # Extract employee ID from selected option
                selected_id = selected_display.split(" - ")[0].strip()
                
                # Get the selected employee's data from the database (more reliable than DataFrame)
                cursor.execute("SELECT * FROM employees WHERE employee_id = ?", (selected_id,))
                employee_db = cursor.fetchone()
                
                if employee_db:
                    # Get database columns
                    # NEW: 10 columns
                    employee_id, employee_name, department, job_title, hire_date, email, phone, photo_path, created_at, is_active = employee_db[:10]
                    
                    # Get user role
                    cursor.execute("SELECT role FROM users WHERE employee_id = ?", (selected_id,))
                    role_result = cursor.fetchone()
                    role = role_result[0] if role_result else "Employee"
                    
                    # Check enrollment status
                    encoding_file = os.path.join(fr_module.ENCODINGS_DIR, f"{selected_id}.pkl")
                    enrolled = "✅" if os.path.exists(encoding_file) else "❌"
                    
                    # Status indicator
                    status = "✅ Active" if is_active == 1 else "❌ Inactive"
                    
                    # Display selected employee info
                    st.markdown(f"""
                    <div style="
                        background: #eeeeee;
                        padding: 1.5rem;
                        border-radius: 10px;
                        color: #000000;  /* Main text to black */
                        margin: 1rem 0;
                        border-left: 5px solid {'#28a745' if status == '✅ Active' else '#dc3545'};
                    ">
                        <h4 style="margin: 0 0 10px 0; color: #000000;">{employee_name}</h4> <!-- Header to black -->
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                            <div>
                                <strong style="color: #000000;">Employee ID:</strong><br> <!-- Label to black -->
                                <span style="font-weight: bold;">{employee_id}</span>
                            </div>
                            <div>
                                <strong style="color: #000000;">Department:</strong><br>
                                <span style="font-weight: bold;">{department or 'N/A'}</span>
                            </div>
                            <div>
                                <strong style="color: #000000;">Job Title:</strong><br>
                                <span style="font-weight: bold;">{job_title or 'N/A'}</span>
                            </div>
                            <div>
                                <strong style="color: #000000;">Role:</strong><br>
                                <span style="font-weight: bold; color: #000000;"> <!-- Value to black -->
                                    {role}
                                </span>
                            </div>
                            <div>
                                <strong style="color: #000000;">Status:</strong><br>
                                <span style="font-weight: bold; color: #000000;">
                                    {status}
                                </span>
                            </div>
                            <div>
                                <strong style="color: #000000;">Face Enrolled:</strong><br>
                                <span style="font-weight: bold; color: #000000;">
                                    {enrolled}
                                </span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Action buttons in columns
                    #col_action1, col_action2, col_action3 = st.columns(3)
                    # Action buttons for selected employee
                    col_action1, col_action2, col_action3 = st.columns(3)

                    with col_action1:
                        if st.button(
                            "✏️ Edit Employee", 
                            key=f"edit_{selected_id}",  # unique per employee
                            use_container_width=True,
                            type="primary"
                        ):
                            # Fetch full employee data from DB
                            cursor.execute('''
                                SELECT employee_id, employee_name, department, job_title, 
                                    hire_date, email, phone, photo_path, created_at, is_active
                                FROM employees 
                                WHERE employee_id = ?
                            ''', (selected_id,))
                            employee_full = cursor.fetchone()
                            
                            if not employee_full:
                                st.error("Employee record not found.")
                                st.stop()

                            # Store employee in session_state for editing
                            st.session_state.edit_mode = True
                            st.session_state.edit_employee_id = selected_id
                            st.session_state.employee_data = {
                                'employee_id': employee_full['employee_id'],
                                'employee_name': employee_full['employee_name'],
                                'department': employee_full['department'],
                                'job_title': employee_full['job_title'],
                                'hire_date': employee_full['hire_date'],
                                'email': employee_full['email'],
                                'phone': employee_full['phone'],
                                'photo_path': employee_full['photo_path'],
                                'created_at': employee_full['created_at'],
                                'is_active': employee_full['is_active']
                            }

                            st.query_params = {"tab": "add_edit", "edit_id": selected_id}
                            st.rerun()

                    # with col_action2:
                    #with col_action2:
                    #with col_action2:
                    #with col_action2:
                    with col_action2:
                        st.markdown("**⚠️ Danger Zone:**")

                        # Step 1 – Arm delete
                        if st.button(
                            "🗑️ Delete Employee",
                            key=f"arm_delete_{selected_id}",
                            use_container_width=True
                        ):
                            st.session_state[f"confirm_delete_{selected_id}"] = True

                        # Step 2 – Confirm
                        if st.session_state.get(f"confirm_delete_{selected_id}", False):
                            st.warning(
                                f"Are you sure you want to permanently delete **{employee_name}**?"
                            )

                            col_yes, col_no = st.columns(2)

                            with col_yes:
                                if st.button(
                                    "✅ YES, DELETE",
                                    key=f"confirm_delete_btn_{selected_id}",
                                    use_container_width=True,
                                    type="primary"
                                ):
                                    handle_delete_employee(selected_id, employee_name)
                                    st.session_state.pop(f"confirm_delete_{selected_id}", None)
                                    st.rerun()

                            with col_no:
                                if st.button(
                                    "❌ Cancel",
                                    key=f"cancel_delete_{selected_id}",
                                    use_container_width=True
                                ):
                                    st.session_state.pop(f"confirm_delete_{selected_id}", None)

                    with col_action3:
                        # Toggle Active Status button
                        current_status = is_active == 1
                        status_text = "Deactivate" if current_status else "Activate"
                        button_type = "secondary" if current_status else "primary"
                        
                        if st.button(f"{status_text} Employee", 
                                key=f"toggle_{selected_id}", 
                                use_container_width=True,
                                type=button_type):
                            
                            new_status = 0 if current_status else 1
                            
                            try:
                                # Update employee status
                                cursor.execute('''
                                    UPDATE employees 
                                    SET is_active = ? 
                                    WHERE employee_id = ?
                                ''', (new_status, selected_id))
                                
                                # Also update user account status
                                cursor.execute('''
                                    UPDATE users 
                                    SET is_active = ? 
                                    WHERE employee_id = ?
                                ''', (new_status, selected_id))
                                
                                conn.commit()
                                st.success(f"✅ Employee {status_text.lower()}d successfully!")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error updating status: {str(e)}")
                        
                    # Quick enrollment action for faces
                    if enrolled == '❌':
                        st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
                        col_enroll1, col_enroll2 = st.columns([3, 1])
                        with col_enroll1:
                            st.warning("⚠️ This employee is not enrolled for face recognition.")
                        with col_enroll2:
                            if st.button("📸 Enroll Face", 
                                    key=f"enroll_{selected_id}", 
                                    use_container_width=True,
                                    type="primary"):
                                # Switch to Tab 3 (Enroll with Face)
                                js = """
                                <script>
                                    // Function to switch to tab 3
                                    function switchToTab3() {
                                        const tabs = window.parent.document.querySelectorAll('[data-testid="stTabs"] button');
                                        if (tabs.length >= 3) {
                                            tabs[2].click();
                                        }
                                    }
                                    // Run after a short delay
                                    setTimeout(switchToTab3, 100);
                                </script>
                                """
                                st.components.v1.html(js, height=0)
                else:
                    st.error("Employee not found in database")
        else:
            st.info("No employees available for actions. Add employees first.")
        
        # xxxxxxxxxxx
        # Add this after the employee actions section
        st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#6078ea;">🔗 Link Users to Employees</h3>', unsafe_allow_html=True)

        # Get users without employee links
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.username, u.employee_name, u.role 
            FROM users u 
            WHERE u.employee_id IS NULL AND u.is_active = 1
        """)
        unlinked_users = cursor.fetchall()

        # Get all employees
        cursor.execute("SELECT employee_id, employee_name FROM employees ORDER BY employee_name")
        all_employees = cursor.fetchall()
        conn.close()

        if unlinked_users:
            st.write(f"Found {len(unlinked_users)} unlinked user accounts:")
            
            for user in unlinked_users:
                username, full_name, role = user
                with st.expander(f"👤 {username} - {full_name} ({role})"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        # Create mapping options
                        employee_options = {f"{emp[0]} - {emp[1]}": emp[0] for emp in all_employees}
                        selected_employee = st.selectbox(
                            f"Select employee for {username}:",
                            options=["Select employee..."] + list(employee_options.keys()),
                            key=f"link_select_{username}"
                        )
                    
                    with col2:
                        if selected_employee != "Select employee...":
                            employee_id = employee_options[selected_employee]
                            if st.button(f"🔗 Link", key=f"link_btn_{username}"):
                                try:
                                    conn = get_db_connection()
                                    cursor = conn.cursor()
                                    cursor.execute("""
                                        UPDATE users 
                                        SET employee_id = ? 
                                        WHERE username = ?
                                    """, (employee_id, username))
                                    conn.commit()
                                    conn.close()
                                    st.success(f"✅ Linked {username} to employee {employee_id}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error: {e}")
        else:
            st.success("✅ All user accounts are linked to employees!")

        # Quick stats
        st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            total = len(df)
            st.metric("Total Employees", total)
        
        with col_stat2:
            active = len(df[df["Status"] == "✅ Active"])
            st.metric("Active", active)
        
        with col_stat3:
            enrolled = len(df[df["Enrolled"] == "✅"])
            st.metric("Face Enrolled", enrolled)
        
        with col_stat4:
            admins = len(df[df["Role"] == "Admin"])
            st.metric("Admins", admins)
    
    else:
        st.info("No employees found matching your filters. Try adjusting your search criteria or add new employees.")
    
    conn.close()

with tab2:
    # -----------------------------
    # Header
    # -----------------------------
    st.markdown('<div id="add-edit-form"></div>', unsafe_allow_html=True)
    if st.session_state.get("edit_mode", False):
        # st.markdown(f'''
        # <div class="edit-mode-header">
        #     <h2>✏️ Editing Employee: {st.session_state.employee_data.get('employee_name', 'Unknown')}</h2>
        #     <p>Employee ID: {st.session_state.employee_data.get('employee_id', 'Unknown')}</p>
        # </div>
        # ''', unsafe_allow_html=True)

        st.markdown(f'''
        <style>
        .edit-mode-header {{
            color: black;
        }}
        .edit-mode-header h2, .edit-mode-header p {{
            color: black; /* Ensures child elements are also black */
            margin-bottom: 5px;
        }}
        </style>
        <div class="edit-mode-header">
            <h2>✏️ Editing Employee: {st.session_state.employee_data.get('employee_name', 'Unknown')}</h2>
            <p>Employee ID: {st.session_state.employee_data.get('employee_id', 'Unknown')}</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('<h3 style="color:#6078ea;">➕ Add New Employee</h3>', unsafe_allow_html=True)        

    # -----------------------------
    # Employee Form
    # -----------------------------
    form_mode = "edit" if st.session_state.get("edit_mode", False) else "add"
    if st.session_state.get("employee_form_mode") != form_mode:
        source_data = st.session_state.employee_data if form_mode == "edit" else {}
        st.session_state.employee_name_input = source_data.get("employee_name", "")
        st.session_state.job_title_input = source_data.get("job_title", "")
        st.session_state.email_input = source_data.get("email", "")
        st.session_state.phone_input = source_data.get("phone", "")
        hire_date_raw = source_data.get("hire_date", "")
        default_hire_date = datetime.now().date()
        if isinstance(hire_date_raw, str) and hire_date_raw:
            try:
                default_hire_date = datetime.strptime(hire_date_raw, "%Y-%m-%d").date()
            except:
                default_hire_date = datetime.now().date()
        st.session_state.hire_date_input = default_hire_date
        st.session_state.employee_form_mode = form_mode

    with st.form("employee_form"):
        col1, col2 = st.columns(2)

        # Column 1
        with col1:
            # Use a simpler approach - no JavaScript needed
            employee_id = st.text_input(
                "Employee ID *",
                value=st.session_state.employee_data.get('employee_id', ''),
                placeholder="EMP001",
                disabled=st.session_state.get("edit_mode", False),
                key="employee_id_input"
            )
            
            employee_name = st.text_input(
                "Full Name *",
                value=st.session_state.get("employee_name_input", ""),
                placeholder="John Doe",
                key="employee_name_input",
            )

            # Department dropdown
            departments = get_departments_from_db() or ["HR", "IT", "Finance", "Sales", "Operations"]
            if st.session_state.get("edit_mode", False):
                current_dept = st.session_state.employee_data.get('department', '')
                default_index = departments.index(current_dept) if current_dept in departments else 0
                if current_dept in departments:
                    st.session_state.department_input = current_dept
            else:
                default_index = 0

            department = st.selectbox(
                "Department *",
                options=departments,
                index=default_index,
                help="Select the employee's department",
                key="department_input",
            )

            job_title = st.text_input(
                "Job Title *",
                value=st.session_state.get("job_title_input", ""),
                placeholder="Software Engineer",
                key="job_title_input",
            )

        # Column 2
        with col2:
            # Hire date
            hire_date_value = st.session_state.employee_data.get('hire_date', '')
            default_date = datetime.now().date()
            if hire_date_value:
                try:
                    if isinstance(hire_date_value, str):
                        default_date = datetime.strptime(hire_date_value, "%Y-%m-%d").date()
                except:
                    default_date = datetime.now().date()
            hire_date = st.date_input("Hire Date *", value=st.session_state.get("hire_date_input", default_date), key="hire_date_input")

            email = st.text_input(
                "Email *",
                value=st.session_state.get("email_input", ""),
                placeholder="john@company.com",
                key="email_input",
            )
            phone = st.text_input(
                "Phone *",
                value=st.session_state.get("phone_input", ""),
                placeholder="+1234567890",
                key="phone_input",
            )

            # Status checkbox
            if st.session_state.get("edit_mode", False):
                is_active = st.checkbox(
                    "Active Employee",
                    value=bool(st.session_state.employee_data.get('is_active', 1))
                )
            else:
                is_active = True

        # -----------------------------
        # Photo upload / display
        # -----------------------------
        uploaded_file = None
        photo_path = None
        st.session_state.setdefault("pending_face_encoding", None)
        st.session_state.setdefault("pending_face_encoding_employee_id", None)
        st.session_state.setdefault("pending_face_encoding_photo_path", None)
        if st.session_state.get("edit_mode", False) and st.session_state.employee_data.get('photo_path'):
            current_photo = st.session_state.employee_data.get('photo_path')
            if current_photo and os.path.exists(current_photo):
                try:
                    img = Image.open(current_photo)
                    st.image(img, width=100, caption="Current Photo", use_container_width=False)
                except:
                    st.info("Current photo exists but cannot be displayed.")

            uploaded_file = st.file_uploader(
                "Change Employee Photo (Optional)",
                type=['jpg','jpeg','png'],
                help="Upload a new photo to replace the existing one",
                key="employee_photo_upload",
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload Employee Photo (Optional)",
                type=['jpg','jpeg','png'],
                help="Upload a clear front-facing photo for face recognition",
                key="employee_photo_upload",
            )

        if uploaded_file is not None:
            effective_employee_id = (
                employee_id
                or st.session_state.employee_data.get("employee_id")
                or st.session_state.get("edit_employee_id")
            )
            photos_dir = "employee_photos"
            os.makedirs(photos_dir, exist_ok=True)
            if not effective_employee_id:
                photo_path = None
                # Skip saving to avoid empty-ID filenames.
                st.session_state.pending_face_encoding = None
                st.session_state.pending_face_encoding_employee_id = None
                st.session_state.pending_face_encoding_photo_path = None
            else:
                image_bytes = uploaded_file.getvalue()
                import hashlib
                photo_hash = hashlib.sha256(image_bytes).hexdigest()
                last_hash = st.session_state.get("last_photo_upload_hash")
                last_emp = st.session_state.get("last_photo_upload_employee_id")
                last_path = st.session_state.get("last_photo_upload_path")
                if last_hash == photo_hash and last_emp == effective_employee_id and last_path:
                    photo_path = last_path
                else:
                    photo_filename = f"{effective_employee_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    photo_path = os.path.join(photos_dir, photo_filename)
            image = Image.open(io.BytesIO(image_bytes)) if effective_employee_id else None
            if image is not None:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                if photo_path:
                    image.save(photo_path, "JPEG", quality=95, optimize=True)
                if effective_employee_id:
                    st.session_state.last_photo_upload_hash = photo_hash
                    st.session_state.last_photo_upload_employee_id = effective_employee_id
                    st.session_state.last_photo_upload_path = photo_path

            # Immediately try to detect and encode a face from the uploaded photo.
            encoding, faces_found = _encode_photo_path(photo_path) if photo_path else (None, 0)
            if encoding is not None:
                st.session_state.pending_face_encoding = encoding
                st.session_state.pending_face_encoding_employee_id = effective_employee_id
                st.session_state.pending_face_encoding_photo_path = photo_path
                st.success(f"Face detected ({faces_found}). Encoding will be saved with the record.")
                
            else:
                st.session_state.pending_face_encoding = None
                st.session_state.pending_face_encoding_employee_id = None
                st.session_state.pending_face_encoding_photo_path = None
                st.warning("No face detected in the uploaded image. Encoding will not be saved.")
        elif st.session_state.get("edit_mode", False) and st.session_state.employee_data.get('photo_path'):
            photo_path = st.session_state.employee_data.get('photo_path')
            # No new upload; do not carry a stale pending encoding forward.
            st.session_state.pending_face_encoding = None
            st.session_state.pending_face_encoding_employee_id = None
            st.session_state.pending_face_encoding_photo_path = None

        # -----------------------------
        # User account fields - SIMPLIFIED VERSION
        # -----------------------------
        if not st.session_state.get("edit_mode", False):
            st.markdown('<h3 style="color:#6078ea;">🔐 User Account Credentials</h3>', unsafe_allow_html=True)
            st.markdown("User accounts are automatically created for each employee.")

            # Show username as employee_id (no input field needed)
            if employee_id:
                st.markdown(f"""
                <div style="
                    background: #f0f2f6;
                    padding: 12px;
                    border-radius: 8px;
                    margin-bottom: 15px;
                    border-left: 4px solid #4CAF50;
                ">
                    <div style="display: flex; align-items: center;">
                        <span style="font-weight: bold; margin-right: 10px;">Username:</span>
                        <code style="background: white; padding: 4px 8px; border-radius: 4px; font-size: 16px;">
                            {employee_id}
                        </code>
                    </div>
                    <div style="font-size: 0.9em; color: #666; margin-top: 5px;">
                        💡 Username automatically set to Employee ID
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            col_pass1, col_pass2 = st.columns(2)
            with col_pass1:
                password = st.text_input("Password *", type="password", key="password_input")
            with col_pass2:
                password_confirm = st.text_input("Confirm Password *", type="password", key="password_confirm_input")
            user_role = st.selectbox("User Role *", options=["Employee", "Manager", "Admin"], index=0, key="user_role_select")
            
        else:
            # EDIT MODE: Show different UI
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT role FROM users WHERE employee_id = ?", (st.session_state.employee_data.get('employee_id'),))
            role_result = cursor.fetchone()
            conn.close()
            if role_result and isinstance(role_result[0], str):
                current_role = role_result[0].capitalize()
            else:
                current_role = "Employee"

            user_role = st.selectbox(
                "User Role",
                options=["Employee", "Manager", "Admin"],
                index=["Employee", "Manager", "Admin"].index(current_role) if current_role in ["Employee","Manager","Admin"] else 0
            )
            
            st.markdown("### 🔐 Update Password (Optional)")
            col_pass1, col_pass2 = st.columns(2)
            with col_pass1:
                new_password = st.text_input("New Password", type="password")
            with col_pass2:
                confirm_password = st.text_input("Confirm New Password", type="password")

        # -----------------------------
        # Form buttons
        # -----------------------------
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submit_button = st.form_submit_button(
                "✅ Save Employee" if st.session_state.get("edit_mode", False) else "✅ Add Employee"
            )
        with col_btn2:
            if st.session_state.get("edit_mode", False):
                cancel_button = st.form_submit_button("❌ Cancel Edit")
                if cancel_button:
                    st.session_state.edit_mode = False
                    st.session_state.edit_employee_id = None
                    st.session_state.employee_data = {}
                    st.query_params = {"tab": "view_employees"}
                    st.rerun()

        # -----------------------------
        # Handle form submission - FIXED VERSION
        # -----------------------------
        if submit_button:
            employee_id = employee_id.strip() if isinstance(employee_id, str) else employee_id
            employee_name = employee_name.strip() if isinstance(employee_name, str) else employee_name
            department = department.strip() if isinstance(department, str) else department
            job_title = job_title.strip() if isinstance(job_title, str) else job_title
            email = email.strip() if isinstance(email, str) else email
            phone = phone.strip() if isinstance(phone, str) else phone

            role_db = user_role.lower() if isinstance(user_role, str) else user_role
            email_pattern = r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$"
            validation_errors = []

            if not employee_id:
                validation_errors.append("❌ Employee ID is required!")
            if not employee_name:
                validation_errors.append("❌ Full Name is required!")
            if not department:
                validation_errors.append("❌ Department is required!")
            if not job_title:
                validation_errors.append("❌ Job Title is required!")
            if hire_date is None:
                validation_errors.append("❌ Hire Date is required!")
            if not email:
                validation_errors.append("❌ Email is required!")
            elif not re.fullmatch(email_pattern, email):
                validation_errors.append("❌ Enter a valid email address (e.g., name@company.com).")
            if not phone:
                validation_errors.append("❌ Phone is required!")
            elif re.search(r"[A-Za-z]", phone):
                validation_errors.append("❌ Phone number cannot contain alphabetic characters.")

            # Validation for new employees
            if not st.session_state.get("edit_mode", False):
                if not password:
                    validation_errors.append("❌ Password is required!")
                if password != password_confirm:
                    validation_errors.append("❌ Passwords do not match!")

            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                st.stop()

            conn = get_db_connection()
            cursor = conn.cursor()
            import hashlib
            import time
            import sqlite3
            try:
                # -----------------------------
                # Employee table
                # -----------------------------
                if st.session_state.get("edit_mode", False):
                    # Edit existing employee
                    cursor.execute('''
                        UPDATE employees
                        SET employee_name=?, department=?, job_title=?, hire_date=?, email=?, phone=?, photo_path=?, is_active=?
                        WHERE employee_id=?
                    ''', (
                        employee_name, department, job_title, hire_date.isoformat(),
                        email, phone, photo_path, int(is_active), employee_id
                    ))
                    action = "updated"
                else:
                    # Add new employee
                    cursor.execute('''
                        INSERT INTO employees (employee_id, employee_name, department, job_title, hire_date, email, phone, photo_path, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (employee_id, employee_name, department, job_title, hire_date.isoformat(), email, phone, photo_path, int(is_active)))
                    action = "added"
                conn.commit()

                # If a new photo was uploaded and encoded, persist the encoding now.
                pending_encoding = st.session_state.get("pending_face_encoding")
                pending_emp_id = st.session_state.get("pending_face_encoding_employee_id")
                pending_photo_path = st.session_state.get("pending_face_encoding_photo_path")
                if (
                    pending_encoding is not None
                    and pending_emp_id == employee_id
                    and pending_photo_path == photo_path
                ):
                    if fr_module.save_encoding(employee_id, employee_name, pending_encoding):
                        st.success(f"✅ Face encoding saved for {employee_name} ({employee_id}).")
                        try:
                            fr_module.load_encodings()
                        except Exception:
                            pass
                    else:
                        st.warning("Employee was saved, but face encoding could not be persisted.")
                    # Clear pending state after attempting to save.
                    st.session_state.pending_face_encoding = None
                    st.session_state.pending_face_encoding_employee_id = None
                    st.session_state.pending_face_encoding_photo_path = None

                # -----------------------------
                # Users table
                # -----------------------------
                if st.session_state.get("edit_mode", False):
                    # Edit mode: update role & optionally password
                    cursor.execute("SELECT id FROM users WHERE employee_id=?", (employee_id,))
                    user_exists = cursor.fetchone()
                    if user_exists:
                        # Update user role, employee_name, email, department
                        cursor.execute('''
                            UPDATE users
                            SET role=?, employee_name=?, email=?, department=?
                            WHERE employee_id=?
                        ''', (role_db, employee_name, email, department, employee_id))

                        # Update password if requested
                        if new_password or confirm_password:
                            if new_password != confirm_password:
                                st.error("❌ Passwords do not match")
                            else:
                                password_hash = hashlib.sha256(new_password.encode()).hexdigest()
                                cursor.execute("UPDATE users SET password_hash=? WHERE employee_id=?", (password_hash, employee_id))
                    else:
                        # If user record missing, create it
                        pw = new_password if new_password else employee_id
                        password_hash = hashlib.sha256(pw.encode()).hexdigest()
                        cursor.execute('''
                            INSERT INTO users (username, password_hash, employee_name, email, role, employee_id, department, created_at, is_active)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            employee_id, password_hash, employee_name, email, role_db, employee_id, department,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), int(is_active)
                        ))
                else:
                    # Add new user for new employee
                    # Always use employee_id as username
                    username_final = employee_id
                    password_hash = hashlib.sha256(password.encode()).hexdigest()
                    
                    cursor.execute('''
                        INSERT INTO users (username, password_hash, employee_name, email, role, employee_id, department, created_at, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        username_final, password_hash, employee_name, email, role_db, employee_id, department,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), int(is_active) 
                    ))

                conn.commit()

                st.success(f"✅ Employee {employee_name} {action} successfully!")
                
                # Show credentials for new employees
                is_new_employee = not st.session_state.get("edit_mode", False)
                if is_new_employee:
                    st.info(f"""
                    **User Account Created:**
                    - **Username:** `{employee_id}`
                    - **Password:** `{password}`
                    - **Role:** `{user_role}`
                    
                    ⚠️ Please save these credentials. Employee should change password on first login.
                    """)

                # -----------------------------
                # Clear form / session state
                # -----------------------------
                st.session_state.edit_mode = False
                st.session_state.edit_employee_id = None
                st.session_state.employee_data = {}

                # Clear form inputs
                for key in ["username", "password", "password_confirm", "new_password", "confirm_password"]:
                    if key in st.session_state:
                        del st.session_state[key]
                if is_new_employee:
                    for key in [
                        "employee_id_input",
                        "employee_name_input",
                        "department_input",
                        "job_title_input",
                        "hire_date_input",
                        "email_input",
                        "phone_input",
                        "employee_photo_upload",
                        "password_input",
                        "password_confirm_input",
                        "user_role_select",
                        "employee_form_mode",
                        "last_photo_upload_hash",
                        "last_photo_upload_employee_id",
                        "last_photo_upload_path",
                        "pending_face_encoding",
                        "pending_face_encoding_employee_id",
                        "pending_face_encoding_photo_path",
                    ]:
                        if key in st.session_state:
                            del st.session_state[key]

                # Switch back to View Employees tab
                st.query_params = {"tab": "view_employees"}

                # Small delay to let UI update, then rerun
                time.sleep(1.5)
                st.rerun()

            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    if "employees.employee_id" in str(e):
                        st.error(f"❌ Employee ID '{employee_id}' already exists!")
                    elif "users.username" in str(e):
                        st.error(f"❌ Username '{employee_id}' already exists!")
                    else:
                        st.error(f"❌ Database error: {str(e)}")
                else:
                    st.error(f"❌ Database error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error saving employee: {str(e)}")
            finally:
                conn.close()

# TAB 3: Enroll Face (Camera) - COMPLETE IMPROVED VERSION
with tab3:
    st.markdown('<h3 style="color:#6078ea;">📸 Enroll Employee Face Capture</h3>', unsafe_allow_html=True)
    
    col_info, col_camera = st.columns([1, 2])
    
    with col_info:
        st.markdown("""
        ### 📋 Instructions:
        
        1. **Select Employee** from the dropdown
        2. **Adjust Camera Settings** if image is dark
        3. **Start Camera** to begin capture
        4. **Position Face** in the frame
        5. **Capture Photos** (3-5 recommended)
        6. **Train Model** to save encodings
        
        ### 💡 Tips:
        - Ensure good, even lighting
        - Remove glasses/sunglasses
        - Look directly at camera
        - Capture different angles
        - Use neutral expression
        """)
        
        # Camera settings for better image quality
        st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#6078ea;">⚙️ Camera Settings</h3>', unsafe_allow_html=True)
        brightness_offset = st.slider(
            "Brightness",
            min_value=-100, max_value=100, value=0,
            help="Adjust image brightness"
        )
        
        contrast_factor = st.slider(
            "Contrast",
            min_value=0.5, max_value=2.0, value=1.0, step=0.1,
            help="Adjust image contrast"
        )
        
        gamma_value = st.slider(
            "Gamma",
            min_value=0.5, max_value=2.0, value=1.0, step=0.1,
            help="Adjust image gamma correction"
        )
        
        # Employee selection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT employee_id, employee_name FROM employees WHERE is_active = 1 ORDER BY employee_name")
        employees = cursor.fetchall()
        conn.close()
        
        if not employees:
            st.warning("No active employees found. Please add employees first.")
            st.session_state.camera_active = False
        else:
            employee_options = {f"{emp[0]} - {emp[1]}": emp[0] for emp in employees}
            selected_employee = st.selectbox(
                "Select Employee to Enroll",
                options=list(employee_options.keys()),
                help="Choose the employee to capture face for"
            )
            
            selected_employee_id = employee_options[selected_employee]
            st.session_state.selected_employee_id = selected_employee_id
            
            # Display current encoding status
            encoding_file = os.path.join(fr_module.ENCODINGS_DIR, f"{selected_employee_id}.pkl")
            if os.path.exists(encoding_file):
                st.success("✅ Face already enrolled")
                st.info("New captures will update the existing encoding")
            else:
                st.warning("⚠️ Face not enrolled yet")
    
    with col_camera:
        # Camera controls
        col_controls1, col_controls2, col_controls3 = st.columns(3)
        
        with col_controls1:
            if st.button("🎬 Start Camera", 
                        use_container_width=True, 
                        type="primary",
                        disabled=not employees):
                st.session_state.camera_active = True
                st.session_state.capture_count = 0
                st.session_state.captured_image = None
                st.session_state.last_frame_bytes = None
                st.rerun()
        
        with col_controls2:
            if st.button("⏹️ Stop Camera", 
                        use_container_width=True, 
                        type="secondary"):
                st.session_state.camera_active = False
                st.session_state.camera_stop_reason = "manual"
                st.rerun()
        
        with col_controls3:
            capture_disabled = not st.session_state.camera_active
            if st.button("📸 Capture Photo", 
                        use_container_width=True, 
                        disabled=capture_disabled):
                if st.session_state.captured_image is not None:
                    # Save the captured frame
                    photos_dir = os.path.join("data", "captured_faces")
                    os.makedirs(photos_dir, exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{selected_employee_id}_capture_{timestamp}.jpg"
                    filepath = os.path.join(photos_dir, filename)
                    
                    # Apply adjustments to captured image
                    adjusted_image = cv2.convertScaleAbs(
                        st.session_state.captured_image, 
                        alpha=contrast_factor, 
                        beta=brightness_offset
                    )
                    
                    if gamma_value != 1.0:
                        adjusted_image = adjust_gamma(adjusted_image, gamma_value)
                    
                    # Convert BGR to RGB before saving
                    rgb_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)
                    
                    # Save with proper quality
                    cv2.imwrite(filepath, rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # Store in BytesIO for enrollment
                    _, buffer = cv2.imencode('.jpg', rgb_image)
                    st.session_state.last_frame_bytes = io.BytesIO(buffer)
                    
                    st.session_state.capture_count += 1
                    st.success(f"✅ Photo {st.session_state.capture_count} captured!")
                    
                    # Show brightness info
                    gray_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
                    mean_brightness = np.mean(gray_image)
                    st.session_state.frame_mean_brightness = mean_brightness
                    st.info(f"📊 Image brightness: {mean_brightness:.1f}/255")
        
        # Camera feed display
        if st.session_state.camera_active:
            st.info("Camera is active. Face detection will run automatically.")
            
            # Initialize webcam with better settings
            cap = cv2.VideoCapture(0)
            
            # Try to set camera properties for better quality
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Higher resolution
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)    # Adjust brightness
                cap.set(cv2.CAP_PROP_CONTRAST, 0.5)      # Adjust contrast
            except:
                pass  # Some cameras don't support these settings
            
            if not cap.isOpened():
                st.error("Cannot access camera. Please check your camera connection.")
                st.session_state.camera_active = False
            else:
                frame_placeholder = st.empty()
                info_placeholder = st.empty()
                brightness_placeholder = st.empty()
                
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("Failed to capture frame")
                        break
                    
                    # Store the current frame for capture
                    st.session_state.captured_image = frame.copy()
                    
                    # Apply adjustments for display
                    adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=brightness_offset)
                    
                    if gamma_value != 1.0:
                        adjusted_frame = adjust_gamma(adjusted_frame, gamma_value)
                    
                    # Calculate and display brightness
                    gray_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2GRAY)
                    mean_brightness = np.mean(gray_frame)
                    brightness_placeholder.text(f"📊 Frame brightness: {mean_brightness:.1f}/255")
                    st.session_state.frame_mean_brightness = mean_brightness
                    
                    # Face detection
                    rgb_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2RGB)
                    face_locations = fr_module.face_recognition.face_locations(rgb_frame)
                    
                    # Draw face rectangles
                    for (top, right, bottom, left) in face_locations:
                        # Draw green rectangle around face
                        cv2.rectangle(adjusted_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        
                        # Add face count text
                        cv2.putText(adjusted_frame, f"Face Detected", (left, top - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Convert to RGB for Streamlit display
                    display_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display the frame
                    frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                    
                    # Display capture info
                    info_placeholder.info(f"**Captures:** {st.session_state.capture_count} | **Faces detected:** {len(face_locations)}")
                    
                    # Small delay
                    time.sleep(0.05)
                
                # Release camera
                cap.release()
                brightness_placeholder.empty()
            
        else:
            # Show placeholder when camera is off
            if st.session_state.camera_stop_reason == "manual":
                st.info("Camera stopped manually.")
            else:
                st.info("👆 Click 'Start Camera' to begin face capture")
            
            # Show last captured image if any
            if st.session_state.capture_count > 0:
                st.markdown(f"**Total Captures:** {st.session_state.capture_count}")
                
                # Show brightness recommendation
                if hasattr(st.session_state, 'frame_mean_brightness'):
                    brightness = st.session_state.frame_mean_brightness
                    if brightness < 100:
                        st.warning(f"⚠️ Image is dark ({brightness:.1f}/255). Try increasing brightness.")
                    elif brightness > 200:
                        st.warning(f"⚠️ Image is too bright ({brightness:.1f}/255). Try decreasing brightness.")
                    else:
                        st.success(f"✅ Good image brightness ({brightness:.1f}/255)")
        
        # Enrollment section
        if st.session_state.capture_count > 0 and st.session_state.last_frame_bytes is not None:
            st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
            st.subheader("🧠 Enroll Face Now")
            
            # Get employee name
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT employee_name FROM employees WHERE employee_id = ?", 
                         (selected_employee_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                employee_name = result[0]
                
                # Show enrollment button
                if st.button("🚀 Enroll Face Now", use_container_width=True, type="primary"):
                    with st.spinner("Enrolling face..."):
                        # Use the captured photo bytes
                        success, result_msg = fr_module.enroll_face(
                            selected_employee_id, 
                            employee_name, 
                            st.session_state.last_frame_bytes
                        )
                        
                        if success:
                            # The result_msg IS the photo_path returned by enroll_face
                            photo_path = result_msg
                            
                            # Update database with photo path
                            conn = get_db_connection()
                            cursor = conn.cursor()
                            cursor.execute('''
                                UPDATE employees 
                                SET photo_path = ? 
                                WHERE employee_id = ?
                            ''', (photo_path, selected_employee_id))
                            conn.commit()
                            conn.close()
                            
                            st.success(f"""
                            ✅ Face enrolled successfully!
                            
                            **Employee:** {employee_name}
                            **Encoding saved to:** {fr_module.ENCODINGS_DIR}/{selected_employee_id}.pkl
                            **Photo saved to:** {photo_path}
                            **Status:** Ready for recognition
                            
                            You can now use this employee for attendance tracking.
                            """)
                            
                            # Verify encoding was saved
                            encoding_file = os.path.join(fr_module.ENCODINGS_DIR, f"{selected_employee_id}.pkl")
                            if os.path.exists(encoding_file):
                                st.info(f"✅ Encoding file created: {encoding_file}")
                                
                                # Try to load and verify the encoding
                                try:
                                    with open(encoding_file, 'rb') as f:
                                        encoding = pickle.load(f)
                                    st.success(f"✓ Encoding loaded successfully (shape: {encoding.shape})")
                                except Exception as e:
                                    st.error(f"✗ Failed to load encoding: {e}")
                            else:
                                st.warning(f"⚠️ Encoding file not found at: {encoding_file}")
                            
                            # Reset
                            st.session_state.capture_count = 0
                            st.session_state.last_frame_bytes = None
                            st.rerun()
                        else:
                            st.error(f"❌ Error enrolling face: {result_msg}")

# TAB 4: Database & Photo Fix Tools
with tab4:
    st.markdown('<h3 style="color:#6078ea;">🔧 Database & Photo Fix Tools</h3>', unsafe_allow_html=True)
    st.caption("Utilities for cleaning up employee photo paths and quick assignment.")

    employees = db.get_all_employees()
    if employees:
        missing_photo_count = 0
        photo_rows = []
        for emp in employees:
            emp_id = emp[0]
            emp_name = emp[1]
            photo_path = emp[7] if len(emp) > 7 else None
            if photo_path:
                photo_rows.append(
                    {
                        "Employee": f"{emp_name} (ID: {emp_id})",
                        "Photo Path": photo_path,
                    }
                )
            else:
                missing_photo_count += 1

        st.markdown(f"##### Employees Without Photos: {missing_photo_count}")
        #st.metric("", missing_photo_count)
        st.markdown("##### Non-Photo Employees") 
        col1, col2 = st.columns([3, 2])

        with col1:
            missing_rows = []
            for emp in employees:
                emp_id = emp[0]
                emp_name = emp[1]
                photo_path = emp[7] if len(emp) > 7 else None
                if not photo_path:
                    missing_rows.append(
                        {
                            "Employee": f"{emp_name} (ID: {emp_id})",
                        }
                    )
            if missing_rows:
                photo_df = pd.DataFrame(missing_rows)
            else:
                photo_df = pd.DataFrame(columns=["Employee"])
            st.dataframe(
                photo_df, 
                use_container_width=True, 
                height=300,
                hide_index=True
            )

    if st.button("🛠️ Fix All Photo Paths", type="secondary"):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            employees = db.get_all_employees()
            fixed_count = 0

            for emp in employees:
                emp_id = emp[0]
                emp_name = emp[1]

                photo_found = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    possible_path = f"employee_photos/{emp_id}{ext}"
                    if os.path.exists(possible_path):
                        photo_found = possible_path
                        break

                if photo_found:
                    cursor.execute(
                        "UPDATE employees SET photo_path = ? WHERE employee_id = ?",
                        (photo_found, emp_id)
                    )
                    fixed_count += 1
                    st.success(f"✓ Fixed {emp_name}: {photo_found}")
                else:
                    st.warning(f"⚠️ No photo found for {emp_name} in employee_photos/ folder")

            conn.commit()
            conn.close()

            if fixed_count > 0:
                st.success(f"✅ Fixed {fixed_count} photo paths!")
                st.info("Please refresh the page to reload face encodings")
                if st.button("🔄 Refresh Now", key="refresh_after_fix"):
                    st.rerun()
            else:
                st.info("No photo paths needed fixing or no photos found")

        except Exception as e:
            st.error(f"Error fixing paths: {e}")

    st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#6078ea;">Employee-Photos Folder Contents</h3>', unsafe_allow_html=True)
    if os.path.exists("employee_photos"):
        photo_files = [f for f in os.listdir("employee_photos") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if photo_files:
            st.info(f"Found {len(photo_files)} photos:")
            cols = st.columns(3)
            for idx, photo_file in enumerate(photo_files):
                with cols[idx % 3]:
                    st.text(photo_file)
        else:
            st.warning("No photos found in Employee-Photos folder")
    else:
        st.error("Employee-Photos folder does not exist!")

    st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#6078ea;">Quick Photo Assignment</h3>', unsafe_allow_html=True)
    if employees:
        selected_emp = st.selectbox(
            "Select Employee:",
            options=[f"{emp[1]} (ID: {emp[0]})" for emp in employees],
            key="fix_emp_select"
        )

        if selected_emp:
            emp_id = selected_emp.split("(ID: ")[1].replace(")", "")

            if os.path.exists("employee_photos"):
                available_photos = [f for f in os.listdir("employee_photos")
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                if available_photos:
                    selected_photo = st.selectbox(
                        "Select Photo:",
                        options=available_photos,
                        key=f"photo_for_{emp_id}"
                    )

                    if st.button(f"Assign Photo to {selected_emp.split(' (')[0]}", key=f"assign_{emp_id}"):
                        new_path = f"employee_photos/{selected_photo}"
                        try:
                            conn = get_db_connection()
                            cursor = conn.cursor()
                            cursor.execute(
                                "UPDATE employees SET photo_path = ? WHERE employee_id = ?",
                                (new_path, emp_id)
                            )
                            conn.commit()
                            conn.close()
                            st.success(f"✅ Assigned {selected_photo} to employee")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("No photos available in employee_photos folder")
                    st.info("Please add photos to the folder and try again.")

# Debug section
with st.expander("🔍 Debug Information"):
    st.subheader("Encoding Files Check")
    
    if st.button("Check Encoding Files"):
        encodings_dir = fr_module.ENCODINGS_DIR
        
        if os.path.exists(encodings_dir):
            encoding_files = os.listdir(encodings_dir)
            
            if encoding_files:
                st.success(f"Found {len(encoding_files)} encoding files in {encodings_dir}:")
                for file in sorted(encoding_files):
                    filepath = os.path.join(encodings_dir, file)
                    size_kb = os.path.getsize(filepath) / 1024
                    st.info(f"📄 {file} ({size_kb:.1f} KB)")
            else:
                st.warning(f"No encoding files found in {encodings_dir}")
        else:
            st.error(f"Encoding directory {encodings_dir} does not exist")
    
    st.subheader("Session State")
    st.json({k: str(v) for k, v in st.session_state.items()})

# Footer
st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>💼 <strong>AI-Powered Attendance System</strong></p>
        <p style="font-size: 0.9em;">Employee Management Module</p>
        <p style="font-size: 0.8em;">Face Enrollment • Employee Records • Database</p>
        <p style="font-size: 0.8em;">Version 2.1 | Developed by: Itoro Udonyah (NOU234244897) | <a href="https://github.com/itoroudonyah" target="_blank">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True) 
