# database.py - UPDATED VERSION WITHOUT REDUNDANT COLUMNS
import sqlite3
import pandas as pd
from datetime import datetime, date, time
import os

# DATABASE_NAME = 'attendance.db'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_NAME = os.path.join(BASE_DIR, "attendance.db")
print("DB PATH:", os.path.abspath(DATABASE_NAME))


def _register_sqlite_adapters():
    """
    Register explicit adapters for Python date/time types.
    This avoids relying on sqlite3's deprecated implicit adapters.
    """
    sqlite3.register_adapter(datetime, lambda dt: dt.isoformat(sep=" "))
    sqlite3.register_adapter(date, lambda d: d.isoformat())
    sqlite3.register_adapter(time, lambda t: t.strftime("%H:%M:%S"))


_register_sqlite_adapters()

def get_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the database and creates all required tables with CORRECT schema.
    """
    conn = get_connection()
    # This line is mandatory for CASCADE to work in SQLite
    conn.execute("PRAGMA foreign_keys = ON;") 
    # return conn
    cursor = conn.cursor()

    # Create employees table with CORRECT schema 
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            employee_name TEXT NOT NULL,
            department TEXT,
            job_title TEXT,
            hire_date TEXT,
            email TEXT,
            phone TEXT,
            photo_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            UNIQUE(employee_id)
        )
    ''')

    # Create attendance table with complete schema (including location columns)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            employee_name TEXT NOT NULL,
            date DATE NOT NULL,
            time TIME NOT NULL,
            check_type TEXT NOT NULL,
            ip_address TEXT,
            location_city TEXT,
            location_region TEXT,
            location_country TEXT,
            latitude TEXT,
            longitude TEXT,
            isp TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
        )
    ''')
    
    # Create anomaly_log table with complete schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS anomaly_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT NOT NULL,
            employee_name TEXT NOT NULL,
            date DATE NOT NULL,
            check_type TEXT NOT NULL,
            anomaly_type TEXT NOT NULL,
            notes TEXT,
            location_city TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
        )
    ''')
    
    # Create users table for authentication
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
            FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
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
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Database initialized successfully: {DATABASE_NAME}")

# Employee Management Functions - UPDATED FOR NEW SCHEMA
def add_employee(employee_id, employee_name, department=None, job_title=None, 
                 hire_date=None, email=None, phone=None, photo_path=None, is_active=True):
    """
    Add a new employee to the database.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO employees (employee_id, employee_name, department, job_title, 
                                  hire_date, email, phone, photo_path, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (employee_id, employee_name, department, job_title, 
              hire_date, email, phone, photo_path, int(is_active)))
        
        conn.commit()
        return True, "Employee added successfully"
    except sqlite3.IntegrityError:
        return False, "Employee ID already exists"
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        conn.close()

def get_employee(employee_id):
    """Get employee details by ID"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM employees WHERE employee_id = ?', (employee_id,))
    employee = cursor.fetchone()
    
    conn.close()
    return employee

def get_all_employees():
    """Get all employees"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT employee_id, employee_name, department, job_title, 
               hire_date, email, phone, photo_path, created_at, is_active
        FROM employees 
        ORDER BY employee_name
    ''')
    
    employees = cursor.fetchall()
    conn.close()
    return employees

def get_all_employees_df():
    """Get all employees as pandas DataFrame (for compatibility)"""
    conn = get_connection()
    df = pd.read_sql_query('''
        SELECT employee_id, employee_name, department, job_title, 
               hire_date, email, phone, photo_path, created_at, is_active
        FROM employees 
        ORDER BY employee_name
    ''', conn)
    conn.close()
    return df

def get_departments():
    """Get unique departments from employees table"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT DISTINCT department FROM employees WHERE department IS NOT NULL AND department != '' ORDER BY department")
        departments = [row[0] for row in cursor.fetchall()]
    except Exception as e:
        print(f"Error fetching departments: {e}")
        departments = []
    
    conn.close()
    return departments

def update_employee(employee_id, **kwargs):
    """Update employee details"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Build update query
    if not kwargs:
        conn.close()
        return False, "No update data provided"
    
    set_clause = ", ".join([f"{key} = ?" for key in kwargs.keys()])
    values = list(kwargs.values())
    values.append(employee_id)
    
    query = f"UPDATE employees SET {set_clause} WHERE employee_id = ?"
    
    try:
        cursor.execute(query, values)
        conn.commit()
        return True, "Employee updated successfully"
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        conn.close()

def delete_employee(employee_id):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # 1. Delete from users table first (Foreign Key requirement)
        cursor.execute("DELETE FROM users WHERE employee_id = ?", (str(employee_id),))
        
        # 2. Delete from attendance records (optional, but recommended)
        cursor.execute("DELETE FROM attendance WHERE employee_id = ?", (str(employee_id),))
        
        # 3. Delete from employees table
        cursor.execute("DELETE FROM employees WHERE employee_id = ?", (str(employee_id),))
        
        conn.commit() # <--- THIS IS CRITICAL
        
        if cursor.rowcount > 0:
            return True, f"Employee {employee_id} deleted successfully."
        else:
            return False, "Employee ID not found in database."
            
    except Exception as e:
        conn.rollback()
        return False, f"Database Error: {str(e)}"
    finally:
        conn.close()

# Attendance Functions
def log_attendance(employee_id, employee_name, check_type, date=None, time=None, **location_data):
    """Log attendance record"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    if time is None:
        time = datetime.now().strftime("%H:%M:%S")
    
    # Prepare location data
    ip_address = location_data.get('ip_address')
    location_city = location_data.get('location_city')
    location_region = location_data.get('location_region')
    location_country = location_data.get('location_country')
    latitude = location_data.get('latitude')
    longitude = location_data.get('longitude')
    isp = location_data.get('isp')
    
    try:
        cursor.execute('''
            INSERT INTO attendance (employee_id, employee_name, date, time, check_type,
                                  ip_address, location_city, location_region, 
                                  location_country, latitude, longitude, isp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (employee_id, employee_name, date, time, check_type,
              ip_address, location_city, location_region,
              location_country, latitude, longitude, isp))
        
        conn.commit()
        return True, "Attendance logged successfully"
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        conn.close()

def get_all_attendance():
    """Get all attendance records"""
    conn = get_connection()
    query = """
        SELECT
            a.id,
            a.employee_id,
            a.employee_name,
            e.department,
            e.job_title,
            a.date,
            a.time,
            a.check_type,
            a.ip_address,
            a.location_city,
            a.location_region,
            a.location_country,
            a.created_at
        FROM attendance a
        LEFT JOIN employees e ON a.employee_id = e.employee_id
        ORDER BY a.date DESC, a.time DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_todays_attendance():
    """Get today's attendance records"""
    conn = get_connection()
    cursor = conn.cursor()
    
    today = datetime.now().strftime("%Y-%m-%d")
    cursor.execute('''
        SELECT * FROM attendance 
        WHERE date = ? 
        ORDER BY time DESC
    ''', (today,))
    
    records = cursor.fetchall()
    conn.close()
    return records

def get_all_users():
    """Get all user accounts from the database"""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT * FROM users
            ORDER BY created_at DESC
        ''')
        users = cursor.fetchall()
        return users
    except Exception as e:
        print(f"Error fetching users: {e}")
        return []
    finally:
        conn.close()
        
def get_employee_attendance(employee_id, start_date=None, end_date=None):
    """Get attendance records for a specific employee"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-01")  # First day of current month
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")  # Today
    
    cursor.execute('''
        SELECT * FROM attendance 
        WHERE employee_id = ? AND date BETWEEN ? AND ?
        ORDER BY date DESC, time DESC
    ''', (employee_id, start_date, end_date))
    
    records = cursor.fetchall()
    conn.close()
    return records

# Anomaly Functions
def log_anomaly(employee_id, employee_name, check_type, anomaly_type, notes=None, location_city=None):
    """Log an attendance anomaly"""
    conn = get_connection()
    cursor = conn.cursor()
    
    date = datetime.now().strftime("%Y-%m-%d")
    
    try:
        cursor.execute('''
            INSERT INTO anomaly_log (employee_id, employee_name, date, check_type, anomaly_type, notes, location_city)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (employee_id, employee_name, date, check_type, anomaly_type, notes, location_city))
        
        conn.commit()
        return True, "Anomaly logged successfully"
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        conn.close()

def get_all_anomalies():
    """Get anomaly records"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM anomaly_log 
        ORDER BY date DESC, created_at DESC
    ''')
    
    anomalies = cursor.fetchall()
    conn.close()
    return anomalies

# User Authentication Functions
def add_user(username, password_hash, employee_name, email=None, role='user', 
             employee_id=None, department=None, is_active=True):
    """Add a new user"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO users (username, password_hash, employee_name, email, role, employee_id, department, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (username, password_hash, employee_name, email, role, employee_id, department, int(is_active)))
        
        conn.commit()
        return True, "User added successfully"
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already exists"
        elif "employee_id" in str(e):
            return False, "Employee ID already linked to another user"
        return False, "Database integrity error"
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        conn.close()

def get_user(username):
    """Get user by username"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, password_hash, employee_name, email, role, employee_id, department, is_active
        FROM users WHERE username = ?
    ''', (username,))
    
    user = cursor.fetchone()
    conn.close()
    return user

def log_login(user_id, username, ip_address=None):
    """Log user login"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO login_logs (user_id, username, login_time, ip_address)
            VALUES (?, ?, datetime('now'), ?)
        ''', (user_id, username, ip_address))
        
        conn.commit()
        login_id = cursor.lastrowid
        return login_id
    except Exception as e:
        print(f"Error logging login: {e}")
        return None
    finally:
        conn.close()

def log_logout(login_id):
    """Log user logout"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            UPDATE login_logs 
            SET logout_time = datetime('now')
            WHERE id = ?
        ''', (login_id,))
        
        conn.commit()
        return True
    except Exception as e:
        print(f"Error logging logout: {e}")
        return False
    finally:
        conn.close()

# Statistics Functions
def get_attendance_stats(date=None):
    """Get attendance statistics for a specific date"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Get total active employees
    cursor.execute('SELECT COUNT(*) FROM employees WHERE is_active = 1')
    total_employees = cursor.fetchone()[0]
    
    # Get check-ins for the day
    cursor.execute('''
        SELECT COUNT(DISTINCT employee_id) FROM attendance 
        WHERE date = ? AND check_type = 'check_in'
    ''', (date,))
    
    checkins = cursor.fetchone()[0]
    
    # Get check-outs for the day
    cursor.execute('''
        SELECT COUNT(DISTINCT employee_id) FROM attendance 
        WHERE date = ? AND check_type = 'check_out'
    ''', (date,))
    
    checkouts = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'total_employees': total_employees,
        'checkins': checkins,
        'checkouts': checkouts,
        'attendance_rate': (checkins / total_employees * 100) if total_employees > 0 else 0
    }

def get_employee_current_status(employee_id, current_date=None):
    """Get current status of an employee for a given date"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if current_date is None:
        current_date = datetime.now().strftime("%Y-%m-%d")
    else:
        current_date = current_date.strftime("%Y-%m-%d")
    
    cursor.execute('''
        SELECT check_type, time 
        FROM attendance 
        WHERE employee_id = ? AND date = ?
        ORDER BY time DESC
        LIMIT 1
    ''', (employee_id, current_date))
    
    last_record = cursor.fetchone()
    conn.close()
    
    if not last_record:
        return "not_checked_in", None, None
    
    last_check_type, last_time = last_record
    
    if last_check_type == 'check_in':
        return "checked_in", last_time, None
    else:
        return "checked_out", None, last_time

# Backup and Maintenance
def backup_database(backup_path=None):
    """Create a backup of the database"""
    if backup_path is None:
        backup_dir = "backups"
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"attendance_backup_{timestamp}.db")
    
    try:
        source_conn = get_connection()
        backup_conn = sqlite3.connect(backup_path)
        
        source_conn.backup(backup_conn)
        
        source_conn.close()
        backup_conn.close()
        
        return True, f"Backup created successfully: {backup_path}"
    except Exception as e:
        return False, f"Backup failed: {str(e)}"

def cleanup_old_records(days_old=90):
    """Clean up old records to keep database size manageable"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cutoff_date = (datetime.now() - pd.Timedelta(days=days_old)).strftime("%Y-%m-%d")
    
    try:
        # Delete old attendance records
        cursor.execute('DELETE FROM attendance WHERE date < ?', (cutoff_date,))
        attendance_deleted = cursor.rowcount
        
        # Delete old anomaly logs
        cursor.execute('DELETE FROM anomaly_log WHERE date < ?', (cutoff_date,))
        anomalies_deleted = cursor.rowcount
        
        # Delete old login logs
        cursor.execute('DELETE FROM login_logs WHERE date(login_time) < ?', (cutoff_date,))
        logs_deleted = cursor.rowcount
        
        conn.commit()
        
        return True, f"Cleaned up: {attendance_deleted} attendance records, {anomalies_deleted} anomalies, {logs_deleted} login logs older than {days_old} days"
    except Exception as e:
        return False, f"Cleanup failed: {str(e)}"
    finally:
        conn.close()

# Initialize database when module is imported
init_db()
print("Database module loaded successfully")
