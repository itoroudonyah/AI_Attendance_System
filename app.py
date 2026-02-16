# app.py
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import io
import time

# Import your modules
import database as db
import face_recognition_module as fr_module

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Attendance System")
st.title("Attendance Monitoring System")

# --- Tabs for Navigation ---
tab_manual, tab_face, tab_students, tab_logs = st.tabs(
    ["Manual Entry", "Face Capture Attendance", "Manage Students", "Attendance Logs"]
)

# --- 1. Manual Entry Tab ---
with tab_manual:
    st.header("Manual Attendance Entry")

    students_df = db.get_all_students()
    if not students_df.empty:
        student_options = {f"{row['name']} ({row['id']})": row['id'] for index, row in students_df.iterrows()}
        selected_student_display = st.selectbox("Select Student:", list(student_options.keys()))

        if selected_student_display:
            selected_student_id = student_options[selected_student_display]
            if st.button("Mark Attendance Manually"):
                db.log_attendance(selected_student_id, "manual")
                st.success(f"Manually marked attendance for {selected_student_display}")
                st.rerun() # Rerun to update logs
        else:
            st.warning("No students registered yet. Please add students in 'Manage Students' tab.")
    else:
        st.info("No students registered. Please register students in the 'Manage Students' tab first.")


# --- 2. Face Capture Attendance Tab ---
with tab_face:
    st.header("Face Capture Attendance")
    st.write("Click 'Start Camera' to capture your face for attendance.")

    # State variable for camera control
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False

    col_cam, col_info = st.columns([2, 1])

    with col_cam:
        if st.button("Start Camera"):
            st.session_state.camera_running = True
        if st.button("Stop Camera"):
            st.session_state.camera_running = False
            # Clear the camera input to stop the stream if it's still active
            # This is a bit tricky with camera_input; sometimes rerun helps
            st.info("Camera stopped.")
            st.rerun()

        frame_placeholder = st.empty() # Placeholder for live camera feed

        if st.session_state.camera_running:
            cap = cv2.VideoCapture(0) # 0 for default webcam

            if not cap.isOpened():
                st.error("Could not open webcam. Please check if it's in use or if drivers are installed.")
                st.session_state.camera_running = False
            else:
                st.write("Camera active. Looking for faces...")
                detected_name = None
                detected_id = None
                last_detection_time = time.time() # To avoid rapid logging

                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to grab frame. Exiting camera stream.")
                        st.session_state.camera_running = False
                        break

                    # Convert BGR to RGB for face_recognition
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Recognize face
                    recognized_id, recognized_name, bbox = fr_module.recognize_face(frame)

                    if recognized_id and recognized_name:
                        # Draw bounding box
                        top, right, bottom, left = bbox
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, recognized_name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

                        if (time.time() - last_detection_time) > 5: # Log attendance every 5 seconds per person
                            db.log_attendance(recognized_id, "face_capture")
                            col_info.success(f"Attendance logged for: **{recognized_name} ({recognized_id})**")
                            last_detection_time = time.time()
                    else:
                        cv2.putText(frame, "Unknown Face", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
                        col_info.info("No recognized face found. Looking...")


                    # Display the frame
                    frame_placeholder.image(frame, channels="BGR", use_column_width=True)
                    time.sleep(0.05) # Small delay to prevent too rapid processing

                cap.release()
                cv2.destroyAllWindows()
            st.session_state.camera_running = False # Ensure state is false after loop ends

    with col_info:
        st.subheader("Attendance Status:")
        # This column will be updated when attendance is logged

# --- 3. Manage Students Tab ---
with tab_students:
    st.header("Register New Student")
    with st.form("new_student_form", clear_on_submit=True):
        student_id = st.text_input("Student ID (Unique):").strip()
        student_name = st.text_input("Student Name:").strip()
        student_class = st.text_input("Class/Grade:").strip()
        uploaded_image = st.camera_input("Capture Photo for Face Enrollment", help="Ensure only one face is visible.")

        submit_button = st.form_submit_button("Add Student")

        if submit_button:
            if student_id and student_name and student_class and uploaded_image:
                # Enroll face first
                success, message_or_path = fr_module.enroll_face(student_id, student_name, uploaded_image)
                if success:
                    photo_path = message_or_path # This will be the saved path
                    # Add student to database
                    db_success = db.add_student(student_id, student_name, student_class, photo_path)
                    if db_success:
                        st.success(f"Student '{student_name}' with ID '{student_id}' added and face enrolled!")
                        fr_module.save_encodings() # Save updated encodings
                    else:
                        st.error(f"Failed to add student '{student_name}'. ID '{student_id}' might already exist.")
                else:
                    st.error(f"Face enrollment failed: {message_or_path}")
            else:
                st.warning("Please fill all student details and capture a photo.")

    st.header("Registered Students")
    current_students_df = db.get_all_students()
    if not current_students_df.empty:
        st.dataframe(current_students_df)
    else:
        st.info("No students registered yet.")

# --- 4. Attendance Logs Tab ---
with tab_logs:
    st.header("Recent Attendance Logs")
    attendance_df = db.get_attendance_logs()
    if not attendance_df.empty:
        st.dataframe(attendance_df)
    else:
        st.info("No attendance records yet.")

# --- Anomaly Detection Placeholder (Future Enhancement) ---
# This part would typically be a separate module (`anomaly_detection_module.py`)
# and might run periodically or on specific triggers.
# For a school project, you can outline its functionality.
# For example:
# with st.sidebar:
#     st.subheader("Anomaly Detection (Future)")
#     st.write("This section will identify unusual attendance patterns.")
#     if st.button("Run Anomaly Detection"):
#         # Placeholder for calling your anomaly detection logic
#         st.info("Running anomaly detection... (Not implemented yet)")
#         # Example: anomalies_df = ad_module.detect_anomalies(db.get_attendance_logs())
#         # st.write(anomalies_df)
