# Take_Attendance.py - Facial Recognition Attendance System
import streamlit as st
import cv2
import numpy as np
from datetime import datetime, date
import time
import os
import sys
import sqlite3
import database as db
from PIL import Image
import tempfile
import pandas as pd
import requests
import json
import pytz
import streamlit.components.v1 as components
from navigation import apply_sidebar_style, render_sidebar, ensure_session, require_roles, render_page_header

st.set_page_config(
    page_title="Take Attendance",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_sidebar_style()
ensure_session(timeout_minutes=None)
render_sidebar("📸 Take Attendance")
require_roles(("admin", "manager", "user", "employee"))

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define Nigeria timezone
NIGERIA_TZ = pytz.timezone('Africa/Lagos')

# Custom CSS
st.markdown("""
<style>
    .camera-feed {
        border: 3px solid #764ba2;
        border-radius: 10px;
        padding: 5px;
        background: #f8f9fa;
    }
    .attendance-log {
        background: #f8f9fa;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .check-in-badge {
        background: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
    .check-out-badge {
        background: #dc3545;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .location-badge {
        background: #4facfe;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
        display: inline-block;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'recognized_employee_id' not in st.session_state:
    st.session_state.recognized_employee_id = None
if 'recognized_employee_name' not in st.session_state:
    st.session_state.recognized_employee_name = None
if 'processing_recognition_action' not in st.session_state:
    st.session_state.processing_recognition_action = False
if 'flash_message' not in st.session_state:
    st.session_state.flash_message = {"type": None, "message": None}
if 'camera_capture' not in st.session_state:
    st.session_state.camera_capture = None
if 'last_attendance_log' not in st.session_state:
    st.session_state.last_attendance_log = None

# Ensure database is initialized
db.init_db()

# --- Flash Message Management ---
if st.session_state.flash_message["type"] is not None:
    # Clear the message after displaying it once (rendered later near manual entry)
    st.session_state.flash_message = {"type": None, "message": None}

# --- Location Capture with Manual Correction ---
st.sidebar.subheader("📍 Location Settings")
st.sidebar.caption(
    "Your location is used to tag attendance records with accurate geographical data. " \
    "Automatic location detection is enabled.",
    # height=68,
    # disabled=True,
) 

# Initialize location in session state
if 'latitude' not in st.session_state:
    st.session_state.latitude = None
if 'longitude' not in st.session_state:
    st.session_state.longitude = None
if 'location_city' not in st.session_state:
    st.session_state.location_city = "Not detected"
if 'location_source' not in st.session_state:
    st.session_state.location_source = None
if 'manual_location_set' not in st.session_state:
    st.session_state.manual_location_set = False

# Abuja coordinates (correct for you)
ABUJA_COORDS = {
    'latitude': 9.0765,
    'longitude': 7.3986,
    'city': 'Abuja, Nigeria'
}

# Function to get location from multiple services
def get_location_from_services():
    services = [
        ('https://ipapi.co/json/', lambda d: (d.get('latitude'), d.get('longitude'), f"{d.get('city', 'Unknown')}, {d.get('country_name', 'Unknown')}")),
        ('https://ipinfo.io/json', lambda d: (float(d.get('loc', '0,0').split(',')[0]) if d.get('loc') else None, 
                                            float(d.get('loc', '0,0').split(',')[1]) if d.get('loc') else None, 
                                            f"{d.get('city', 'Unknown')}, {d.get('country', 'Unknown')}")),
        ('https://geolocation-db.com/json/', lambda d: (d.get('latitude'), d.get('longitude'), f"{d.get('city', 'Unknown')}, {d.get('country_name', 'Unknown')}"))
    ]
    
    for url, parser in services:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                lat, lon, city = parser(data)
                if lat and lon:
                    return lat, lon, city, url
        except:
            continue
    
    return None, None, "Unable to detect", None

# Auto-detect location on first load
if not st.session_state.location_source and not st.session_state.manual_location_set:
    # Use st.spinner (global) instead of st.sidebar.spinner
    with st.spinner("Detecting location..."):
        lat, lon, city, source = get_location_from_services()
        
        if lat and lon:
            st.session_state.latitude = lat
            st.session_state.longitude = lon
            st.session_state.location_city = city
            st.session_state.location_source = "Auto-detected"
            # This part is fine: you can put the success message in the sidebar
            st.sidebar.success(f"📍 {city}")
        else:
            # Use Abuja as default
            st.session_state.latitude = ABUJA_COORDS['latitude']
            st.session_state.longitude = ABUJA_COORDS['longitude']
            st.session_state.location_city = ABUJA_COORDS['city']
            st.session_state.location_source = "Default (Abuja)"
            st.sidebar.info("📍 Using Abuja as default")

# Manual location correction
st.sidebar.markdown("---")
st.sidebar.markdown("**Correct Location if Wrong:**")

# Quick location buttons
col_quick1, col_quick2 = st.sidebar.columns(2)
with col_quick1:
    if st.button("🏙️ Abuja", key="set_abuja", help="Set to Abuja, Nigeria"):
        st.session_state.latitude = ABUJA_COORDS['latitude']
        st.session_state.longitude = ABUJA_COORDS['longitude']
        st.session_state.location_city = ABUJA_COORDS['city']
        st.session_state.location_source = "Manual (Abuja)"
        st.session_state.manual_location_set = True
        st.sidebar.success("Set to Abuja!")
        st.rerun()

with col_quick2:
    if st.button("🔄 Redetect", key="redetect_loc", help="Try detecting again"):
        st.session_state.latitude = None
        st.session_state.longitude = None
        st.session_state.location_city = "Detecting..."
        st.session_state.location_source = None
        st.session_state.manual_location_set = False
        st.rerun()

# Manual coordinate input
with st.sidebar.expander("📝 Manual Coordinates", expanded=False):
    manual_lat = st.number_input("Latitude", value=9.0765, format="%.4f", key="manual_lat")
    manual_lon = st.number_input("Longitude", value=7.3986, format="%.4f", key="manual_lon")
    manual_city = st.text_input("City Name", value="Abuja, Nigeria", key="manual_city")
    
    if st.button("✅ Set Manual Location", key="set_manual"):
        st.session_state.latitude = manual_lat
        st.session_state.longitude = manual_lon
        st.session_state.location_city = manual_city
        st.session_state.location_source = f"Manual: {manual_city}"
        st.session_state.manual_location_set = True
        st.sidebar.success(f"Set to {manual_city}!")
        st.rerun()

# Display current location info
st.sidebar.markdown("---")
if st.session_state.latitude and st.session_state.longitude:
    st.sidebar.markdown(f"**📍 {st.session_state.location_city}**")
    st.sidebar.caption(f"Lat: {st.session_state.latitude:.4f}")
    st.sidebar.caption(f"Lon: {st.session_state.longitude:.4f}")
    if st.session_state.location_source:
        st.sidebar.caption(f"Source: {st.session_state.location_source}")
    
    # Show if location might be inaccurate
    if "Kano" in st.session_state.location_city and not st.session_state.manual_location_set:
        st.sidebar.warning("⚠️ Location may be inaccurate. Click 'Abuja' above to correct.")
else:
    st.sidebar.info("📍 Location detection in progress...")

# Use these variables in your code
latitude = st.session_state.latitude
longitude = st.session_state.longitude
location_city = st.session_state.location_city

# --- Face Recognition Setup ---
@st.cache_resource
def load_known_faces():
    """Loads employee face encodings from the encodings folder."""
    try:
        import face_recognition_module as fr_module
        return fr_module.load_encodings()
    except Exception as e:
        st.error(f"❌ Failed to load face encodings: {e}")
        return [], [], []

# Helper function to convert time string to datetime.time
def parse_time_string(time_str):
    """Convert time string to datetime.time object."""
    if time_str is None:
        return None
    try:
        # Try parsing with seconds
        return datetime.strptime(time_str, "%H:%M:%S").time()
    except ValueError:
        try:
            # Try parsing without seconds
            return datetime.strptime(time_str, "%H:%M").time()
        except ValueError:
            return None

# Function to initialize and manage camera
def initialize_camera():
    """Initialize camera with error handling."""
    try:
        # Try different camera indices (0, 1, 2)
        for camera_index in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                # Test if we can read a frame
                ret, test_frame = cap.read()
                if ret:
                    # Set reasonable resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    return cap, camera_index
                cap.release()
        return None, None
    except Exception as e:
        st.error(f"Camera initialization error: {e}")
        return None, None

# Header
render_page_header("📸 Take Attendance")

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<h3 style="color:#6078ea;">🎥 Real-time Face Recognition</h3>', unsafe_allow_html=True)
    
    # First, check if employee_photos folder exists
    if not os.path.exists("employee_photos"):
        st.error("""
        ❌ **employee_photos folder not found!**
        
        Please create a folder called `employee_photos` in your project directory
        and add employee photos there.
        
        Photos should be named like: `EMP001.jpg`, `EMP002.jpg`, etc.
        """)
        st.session_state.camera_active = False
    else:
        # Check what's in the folder
        photo_files = [f for f in os.listdir("employee_photos") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not photo_files:
            st.warning("""
            ⚠️ **No photos found in employee_photos folder!**
            
            Please add employee photos to the `employee_photos` folder.
            Photos should be named like: `EMP001.jpg`, `EMP002.jpg`, etc.
            """)
        else:
            st.info(f"📸 Found {len(photo_files)} photos in employee_photos folder")
    
    # Load known faces
    # with st.spinner("Loading face encodings..."):
    known_face_encodings, known_face_names, known_face_ids = load_known_faces()

    # Check if face encodings are available
    
    if len(known_face_encodings) == 0:
        st.error("""
        ⚠️ **No face encodings could be loaded!**
        
        **Common issues:**
        1. **Photos not found** - Check if photos exist in `employee_photos/` folder
        2. **Wrong photo names** - Photos should be named like `EMP001.jpg`
        3. **Database paths wrong** - Use the fix tool below to correct paths
        4. **No faces in photos** - Ensure photos show clear frontal faces
        
        **Quick fix:**
        1. Go to **🔧 Database & Photo Fix Tools** section below
        2. Click **🛠️ Fix All Photo Paths** button
        3. Refresh the page
        """)
        st.session_state.camera_active = False
    else:
        st.success(f"✅ Successfully loaded {len(known_face_encodings)} face encodings")

    # Rest of your existing camera code here...
    # Control buttons, camera display, etc.

    # Control buttons
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("🎬 Start Camera Feed", use_container_width=True, type="primary",
                    disabled=len(known_face_encodings) == 0):
            st.session_state.camera_active = True
            st.rerun()
    
    with col_btn2:
        if st.session_state.camera_active:
            if st.button("⏹️ Stop Camera Feed", use_container_width=True, type="secondary"):
                st.session_state.camera_active = False
                st.session_state.recognized_employee_id = None
                st.session_state.recognized_employee_name = None
                st.session_state.processing_recognition_action = False
                # Release camera if exists
                if st.session_state.camera_capture:
                    st.session_state.camera_capture.release()
                    st.session_state.camera_capture = None
                st.rerun()
        else:
            st.button("⏹️ Stop Camera Feed (Inactive)", use_container_width=True, disabled=True)
    
    # Display camera feed or placeholder
    if st.session_state.camera_active and len(known_face_encodings) > 0:
        try:
            import face_recognition
            
            frame_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Initialize camera if not already done
            if st.session_state.camera_capture is None:
                with st.spinner("Initializing camera..."):
                    cap, camera_index = initialize_camera()
                    if cap is None:
                        st.error("❌ Cannot access any camera. Please check:")
                        st.markdown("""
                        1. **Camera is connected** and turned on
                        2. **No other app** is using the camera
                        3. **Browser permissions** allow camera access
                        4. **macOS users**: Check Security & Privacy → Camera
                        5. **Try refreshing** the page and restarting camera
                        """)
                        st.session_state.camera_active = False
                        st.rerun()
                    else:
                        st.session_state.camera_capture = cap
                        st.session_state.camera_index = camera_index
                        st.info(f"✅ Camera {camera_index} initialized successfully")
            
            # Get camera object
            cap = st.session_state.camera_capture
            
            if cap and cap.isOpened():
                st.info("Camera is active. Look at the camera for recognition.")
                
                while st.session_state.camera_active and not st.session_state.recognized_employee_id:
                    try:
                        ret, frame = cap.read()
                        
                        if not ret:
                            st.error("⚠️ Failed to capture frame. Camera might be disconnected.")
                            st.session_state.camera_active = False
                            if cap:
                                cap.release()
                                st.session_state.camera_capture = None
                            st.rerun()
                            break
                        
                        # Convert frame from BGR to RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Find all faces in frame
                        face_locations = face_recognition.face_locations(rgb_frame)
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                        
                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                            name = "Unknown"
                            employee_id = None
                            
                            # Find the best match
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            
                            if matches[best_match_index] and face_distances[best_match_index] < 0.5:
                                name = known_face_names[best_match_index]
                                employee_id = known_face_ids[best_match_index]
                                
                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                            cv2.putText(frame, name, (left + 6, bottom - 6), 
                                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                            
                            if employee_id:
                                # Store detected user in session state
                                st.session_state.recognized_employee_id = employee_id
                                st.session_state.recognized_employee_name = name
                                st.session_state.camera_active = False
                                # Don't release camera yet, we might need it again
                                st.rerun()
                                break
                        
                        # Display the frame
                        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
                        
                        # Small delay to prevent high CPU usage
                        time.sleep(0.1)
                        
                    except Exception as e:
                        st.error(f"Camera error: {e}")
                        st.session_state.camera_active = False
                        if cap:
                            cap.release()
                            st.session_state.camera_capture = None
                        st.rerun()
                        break
                
        except ImportError:
            st.error("Face recognition library not available. Please install with: pip install face-recognition")
            st.session_state.camera_active = False
    
    elif not st.session_state.camera_active:
        st.info("👆 Click 'Start Camera Feed' to begin facial recognition")
    
    # --- Section for recognized employee action ---
    if st.session_state.recognized_employee_id and not st.session_state.processing_recognition_action:
        st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
        st.subheader("✅ Employee Recognized")
        
        employee_id = st.session_state.recognized_employee_id
        employee_name = st.session_state.recognized_employee_name
        
        current_datetime_tz = datetime.now(NIGERIA_TZ)
        current_date_naive = current_datetime_tz.date()
        current_time_naive = current_datetime_tz.replace(tzinfo=None)
        
        # Get employee status
        status, last_check_in_time_str, last_check_out_time_str = db.get_employee_current_status(employee_id, current_date_naive)
        
        # Parse time strings to datetime.time objects
        last_check_in_time = parse_time_string(last_check_in_time_str)
        last_check_out_time = parse_time_string(last_check_out_time_str)
        
        if status == "checked_in":
            st.session_state.action_check_type = "check_out"
            badge_class = "check-out-badge"
            badge_text = "CHECK-OUT"
            if last_check_in_time:
                status_message = f"**{employee_name}** is currently **CHECKED IN** since {last_check_in_time.strftime('%H:%M:%S')}"
            else:
                status_message = f"**{employee_name}** is currently **CHECKED IN**"
            action_button_label = "Record Check-OUT"
        elif status == "checked_out":
            st.session_state.action_check_type = "check_in"
            badge_class = "check-in-badge"
            badge_text = "CHECK-IN"
            status_message = f"**{employee_name}** is ready for **CHECK-IN**"
            action_button_label = "Record Check-IN"
        else:  # not_checked_in
            st.session_state.action_check_type = "check_in"
            badge_class = "check-in-badge"
            badge_text = "CHECK-IN"
            status_message = f"**{employee_name}** is ready for **CHECK-IN**"
            action_button_label = "Record Check-IN"
        
        # Display status
        st.markdown(f"""
        <div class="status-card" style="background-color: white; color: black;">
            <h3>👤 Employee: {employee_name}</h3>
            <p><strong>Employee ID:</strong> {employee_id}</p>
            <p><strong>Status:</strong> <span class="{badge_class[1:]}">{badge_text}</span></p>
            <p>{status_message}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons
        col_action1, col_action2 = st.columns(2)
        
        with col_action1:
            if st.button(action_button_label, use_container_width=True, type="primary"):
                st.session_state.processing_recognition_action = True
                
                with st.spinner(f"Recording {st.session_state.action_check_type.replace('_',' ').upper()}..."):
                    # Use log_attendance instead of take_attendance
                    success, message = db.log_attendance(
                        employee_id,
                        employee_name,
                        st.session_state.action_check_type,
                        current_time=current_time_naive.strftime("%H:%M:%S"),
                        latitude=str(latitude) if latitude else None,
                        longitude=str(longitude) if longitude else None
                    )
                    
                    if success:
                        st.session_state.flash_message = {"type": "success", "message": message}
                        st.session_state.last_attendance_log = {
                            "employee_id": employee_id,
                            "employee_name": employee_name,
                            "check_type": st.session_state.action_check_type,
                        }
                    else:
                        st.session_state.flash_message = {"type": "error", "message": message}
                
                # Reset states
                st.session_state.recognized_employee_id = None
                st.session_state.recognized_employee_name = None
                st.session_state.processing_recognition_action = False
                st.session_state.camera_active = False
                # Release camera
                if st.session_state.camera_capture:
                    st.session_state.camera_capture.release()
                    st.session_state.camera_capture = None
                st.rerun()
        
        with col_action2:
            if st.button("Cancel & Restart Camera", use_container_width=True, type="secondary"):
                st.session_state.recognized_employee_id = None
                st.session_state.recognized_employee_name = None
                st.session_state.processing_recognition_action = False
                st.session_state.camera_active = True
                st.rerun()

with col2:
    st.markdown('<h3 style="color:#6078ea;">📋 Attendance Information</h3>', unsafe_allow_html=True)
    # Current status
    with st.container():
        st.markdown('<div class="status-card">', unsafe_allow_html=True)
        #st.markdown("### 📅 Today's Date")
        st.markdown('<h3 style="color:#6078ea;">📅 Today\'s Date</h3>', unsafe_allow_html=True)
        current_date = date.today()
        st.markdown(f"**{current_date.strftime('%A, %B %d, %Y')}**")
        st.markdown(f"**Current Time:** {datetime.now().strftime('%H:%M:%S')}")
        
        # Show location information
        if latitude is not None and longitude is not None:
            #st.markdown("### 📍 Location Information")
            st.markdown('<h3 style="color:#6078ea;">📍 Location Information</h3>', unsafe_allow_html=True)
            st.markdown(f"**Latitude:** {latitude:.4f}")
            st.markdown(f"**Longitude:** {longitude:.4f}")
            st.markdown(f"**Timezone:** Africa/Lagos")
        elif location_error:
            #st.markdown("### 📍 Location Information")
            st.markdown('<h3 style="color:#6078ea;">### 📍 Location Information</h3>', unsafe_allow_html=True)
            st.warning(f"Location Error: {location_error}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Quick stats
    #st.subheader("📈 Today's Summary")
    st.markdown('<h3 style="color:#6078ea;">📈 Today\'s Summary</h3>', unsafe_allow_html=True)
    
    # Get attendance statistics
    try:
        stats = db.get_attendance_stats()
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Checked In", stats['checkins'])
        with col_stat2:
            st.metric("Checked Out", stats['checkouts'])
        
        attendance_rate = stats['attendance_rate']
        st.progress(attendance_rate / 100, text=f"Attendance Rate: {attendance_rate:.1f}%")
        
    except Exception as e:
        st.error(f"Error fetching stats: {e}")
    
    # Instructions
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Start Camera** - Click the 'Start Camera Feed' button
        2. **Position Face** - Ensure your face is clearly visible
        3. **Automatic Detection** - System will recognize your face automatically
        4. **Confirm Action** - Review your status and click the appropriate button
        5. **Location Tracking** - Your location is automatically recorded
        6. **Stop Camera** - Click 'Stop Camera' when done
        
        ### Features:
        - ✅ **Smart check-in/check-out** based on your daily status
        - 📍 **Location tracking** with GPS coordinates
        - 📊 **Real-time face recognition**
        - 🔒 **Secure and automated** logging
        - 📱 **Mobile-friendly** interface
        
        ### Notes:
        - Ensure good lighting for better accuracy
        - Remove sunglasses or face coverings
        - Look directly at the camera
        - One person at a time for accurate logging
        - Location data helps verify attendance authenticity
        """)

# --- Manual Attendance Entry Section ---
st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
if st.session_state.last_attendance_log: 
    info = st.session_state.last_attendance_log
    check_type = info["check_type"].replace("_", " ").title()
    st.success(
        f"✅ {check_type} logged for {info['employee_name']} (ID: {info['employee_id']})."
    )
    st.session_state.last_attendance_log = None
#st.subheader("🖊️ Manual Attendance Entry (Backup)")
st.markdown('<h3 style="color:#6078ea;">🖊️ Manual Attendance Entry (Backup)</h3>', unsafe_allow_html=True)
st.warning("This option is for situations where facial recognition is not feasible (e.g., camera malfunction). Use responsibly.")

employees = db.get_all_employees()
if employees:
    # Create dictionary for employee selection
    employee_names_ids = {}
    for employee in employees:
        employee_id = employee[0]
        employee_name = employee[1]
        employee_names_ids[f"{employee_name} (ID: {employee_id})"] = employee_id
    
    if employee_names_ids:
        selected_employee_display = st.selectbox("Select Employee for Manual Entry:", 
                                                 options=list(employee_names_ids.keys()),
                                                 index=0 if employee_names_ids else None,
                                                 key="manual_employee_select")
        
        if selected_employee_display:
            selected_employee_id = employee_names_ids[selected_employee_display]
            
            # Find employee name
            selected_employee_name = ""
            for employee in employees:
                if employee[0] == selected_employee_id:
                    selected_employee_name = employee[1]
                    break
            
            manual_current_datetime_tz = datetime.now(NIGERIA_TZ)
            manual_status, manual_last_check_in_time_str, manual_last_check_out_time_str = db.get_employee_current_status(
                selected_employee_id, manual_current_datetime_tz.date()
            )
            
            # Parse time strings
            manual_last_check_in_time = parse_time_string(manual_last_check_in_time_str)
            manual_last_check_out_time = parse_time_string(manual_last_check_out_time_str)
            
            status_message_parts = [
                f"**Current Status for {selected_employee_display}:**",
                f"**{manual_status.replace('_',' ').upper()}**"
            ]
            
            if manual_last_check_in_time:
                status_message_parts.append(f" (Last In: {manual_last_check_in_time.strftime('%H:%M:%S')})")
            
            if manual_last_check_out_time:
                status_message_parts.append(f" (Last Out: {manual_last_check_out_time.strftime('%H:%M:%S')})")
            
            st.info(" ".join(status_message_parts))
            
            col1_manual, col2_manual = st.columns(2)
            
            with col1_manual:
                if manual_status == "checked_in":
                    default_manual_check_type = "Check-OUT"
                else:
                    default_manual_check_type = "Check-IN"
                
                manual_check_type = st.radio("Select Manual Entry Type:", 
                                             ["Check-IN", "Check-OUT"],
                                             index=0 if default_manual_check_type == "Check-IN" else 1,
                                             key="manual_check_type_radio")
            
            with col2_manual:
                manual_time = st.time_input("Manual Time (HH:MM):", manual_current_datetime_tz.time(), key="manual_time_input")
                manual_date = st.date_input("Manual Date:", manual_current_datetime_tz.date(), key="manual_date_input")
            
            manual_datetime_naive = datetime.combine(manual_date, manual_time)
            
            if st.button("Record Manual Attendance", type="secondary", use_container_width=True):
                with st.spinner("Recording manual attendance..."):
                    # Use log_attendance function from database
                    success, message = db.log_attendance(
                        selected_employee_id,
                        selected_employee_name,
                        manual_check_type.replace('-', '_').lower(),
                        date=manual_date.strftime("%Y-%m-%d"),
                        time=manual_time.strftime("%H:%M:%S"),
                        latitude=str(latitude) if latitude else None,
                        longitude=str(longitude) if longitude else None
                    )
                    
                    if success:
                        st.session_state.flash_message = {"type": "success", "message": message}
                    else:
                        st.session_state.flash_message = {"type": "error", "message": message}
                
                st.rerun()
    else:
        st.info("No employees available for manual entry. Please add employees first.")
else:
    st.info("No employees registered. Please add employees first.")

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

# Cleanup when page is closed
if not st.session_state.camera_active and st.session_state.camera_capture:
    st.session_state.camera_capture.release()
    st.session_state.camera_capture = None
