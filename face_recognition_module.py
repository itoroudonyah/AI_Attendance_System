# face_recognition_module.py - IMPROVED WITH BETTER CAMERA HANDLING
import face_recognition
import numpy as np
import os
import pickle
import database as db
import cv2

ENCODINGS_DIR = 'encodings'
if not os.path.exists(ENCODINGS_DIR):
    os.makedirs(ENCODINGS_DIR)

# Global variables to store loaded encodings and names
known_face_encodings = []
known_face_names = []
known_employee_ids = []

def load_encodings():
    """Loads all known face encodings and their corresponding employee IDs and names."""
    global known_face_encodings, known_face_names, known_employee_ids
    known_face_encodings = []
    known_face_names = []
    known_employee_ids = []

    # Get employees from database
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Use the correct column name
    cursor.execute("SELECT employee_id, employee_name, photo_path FROM employees WHERE is_active = 1")
    employees = cursor.fetchall()
    conn.close()

    if not employees:
        print("No employees found in the database. Face recognition will not work.")
        return [], [], []

    for employee in employees:
        employee_id = employee[0]
        employee_name = employee[1]
        photo_path = employee[2]

        if photo_path and os.path.exists(photo_path):
            encoding_path = os.path.join(ENCODINGS_DIR, f"{employee_id}.pkl")
            if os.path.exists(encoding_path):
                try:
                    with open(encoding_path, 'rb') as f:
                        encoding = pickle.load(f)
                        known_face_encodings.append(encoding)
                        known_face_names.append(employee_name)
                        known_employee_ids.append(employee_id)
                except Exception as e:
                    print(f"Error loading encoding for {employee_id}: {e}")
            else:
                # If encoding file doesn't exist, try to generate it from photo_path
                try:
                    image = face_recognition.load_image_file(photo_path)
                    face_encodings = face_recognition.face_encodings(image)
                    if face_encodings:
                        encoding = face_encodings[0]
                        with open(encoding_path, 'wb') as f:
                            pickle.dump(encoding, f)
                        known_face_encodings.append(encoding)
                        known_face_names.append(employee_name)
                        known_employee_ids.append(employee_id)
                    else:
                        print(f"No face found in photo for employee: {employee_name} ({employee_id})")
                except Exception as e:
                    print(f"Error processing photo for employee {employee_name} ({employee_id}): {e}")
        else:
            print(f"Photo path not found or invalid for employee: {employee_name} ({employee_id})")

    print(f"Loaded {len(known_face_encodings)} face encodings.")
    return known_face_encodings, known_face_names, known_employee_ids

def save_encoding(employee_id, employee_name, encoding):
    """
    Save face encoding for an employee.
    
    Args:
        employee_id: Unique employee ID
        employee_name: Employee's full name
        encoding: Face encoding array
    """
    try:
        # Create encodings directory if it doesn't exist
        os.makedirs(ENCODINGS_DIR, exist_ok=True)
        
        # Save to file
        filename = f"{employee_id}.pkl"
        filepath = os.path.join(ENCODINGS_DIR, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(encoding, f)
        
        print(f"Face encoding saved for {employee_name} (ID: {employee_id})")
        
        # Update global variables
        if not any(np.array_equal(encoding, existing) for existing in known_face_encodings):
            known_face_encodings.append(encoding)
            known_face_names.append(employee_name)
            known_employee_ids.append(employee_id)
        
        return True
    except Exception as e:
        print(f"Error saving encoding for {employee_id}: {e}")
        return False

def enroll_face(employee_id, employee_name, photo_bytes):
    """
    Enrolls a new employee's face into the system with brightness adjustment.
    """
    if not photo_bytes:
        return False, "No photo provided for enrollment."

    try:
        # Convert BytesIO object to numpy array
        np_arr = np.frombuffer(photo_bytes.getvalue(), np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Apply brightness/contrast adjustment if image is dark
        if np.mean(image) < 100:  # If image is dark
            # Apply brightness and contrast adjustment
            alpha = 1.3  # Contrast
            beta = 30    # Brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Convert BGR to RGB as face_recognition expects RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return False, "No face found in the captured photo. Please try again with a clear photo."

        # Take the first face found
        face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]

        # Save the encoding
        save_encoding(employee_id, employee_name, face_encoding)

        # Save the photo itself for later reference
        photos_dir = 'employee_photos'
        if not os.path.exists(photos_dir):
            os.makedirs(photos_dir)
        photo_filename = f"{employee_id}.jpg"
        photo_path = os.path.join(photos_dir, photo_filename)
        
        # Save with enhanced quality
        cv2.imwrite(photo_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return True, photo_path

    except IndexError:
        return False, "No face found for encoding. Please ensure a clear face is visible."
    except Exception as e:
        return False, f"An error occurred during enrollment: {e}"

def recognize_face(frame, brightness_offset=0, contrast_factor=1.0, gamma_value=1.0):
    """
    Recognizes a face in the given frame with adjustable brightness.
    
    Args:
        frame: Input frame from camera
        brightness_offset: Brightness adjustment (-200 to 200)
        contrast_factor: Contrast adjustment (0.1 to 3.0)
        gamma_value: Gamma adjustment (0.1 to 2.5)
    """
    if not known_face_encodings:
        print("No known face encodings loaded. Face recognition will not work.")
        load_encodings()
        if not known_face_encodings:
            return False, None, None

    # Apply image adjustments
    adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=brightness_offset)
    
    # Apply gamma correction
    if gamma_value != 1.0:
        inv_gamma = 1.0 / gamma_value
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        adjusted_frame = cv2.LUT(adjusted_frame, table)

    # Convert the adjusted BGR image to RGB
    rgb_frame = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2RGB)

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        # Compare the current face encoding with all known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        
        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            employee_id = known_employee_ids[best_match_index]
            employee_name = known_face_names[best_match_index]
            return True, employee_id, employee_name

    return False, None, None

# Initialize encodings when the module is loaded
db.init_db()
load_encodings()
