"""
Shared sidebar navigation for all pages.
Hides Streamlit's default page list and links to actual page files.
"""

from datetime import datetime, timedelta
from typing import Optional
import uuid
import streamlit as st
import database as db

NAV_ITEMS = [
    {"label": "🏠 Dashboard", "page": "pages/0_Dashboard.py", "roles": ("admin", "manager", "user", "employee")},
    {"label": "📸 Take Attendance", "page": "pages/Take_Attendance.py", "roles": ("admin", "manager", "user", "employee")},
    {"label": "👤 My Attendance", "page": "pages/My_Attendance.py", "roles": ("admin", "manager", "user", "employee")},
    {"label": "👥 Manage Employees", "page": "pages/1_Manage_Employees.py", "roles": ("admin",)},
    {"label": "📊 View Records", "page": "pages/View_Records.py", "roles": ("admin", "manager")},
    {"label": "🚨 Anomaly Detection", "page": "pages/Anomaly_Detection.py", "roles": ("admin", "manager")},
    {"label": "📈 Anomaly Visuals", "page": "pages/Anomaly_Visuals.py", "roles": ("admin", "manager")},
    {"label": "🧪 Synthetic Data", "page": "pages/2_Synthetic_Data.py", "roles": ("admin",)},
    {"label": "⚙️ System Settings", "page": "pages/System_Settings.py", "roles": ("admin",)},
]

def _init_session_table() -> None:
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_sessions (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            role TEXT NOT NULL,
            employee_name TEXT,
            employee_id TEXT,
            department TEXT,
            last_active TEXT NOT NULL,
            remember_me INTEGER NOT NULL DEFAULT 0,
            expiry_minutes INTEGER NOT NULL DEFAULT 5
        )
        """
    )
    cursor.execute("PRAGMA table_info(user_sessions)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if "remember_me" not in existing_cols:
        cursor.execute("ALTER TABLE user_sessions ADD COLUMN remember_me INTEGER NOT NULL DEFAULT 0")
    if "expiry_minutes" not in existing_cols:
        cursor.execute("ALTER TABLE user_sessions ADD COLUMN expiry_minutes INTEGER NOT NULL DEFAULT 5")
    conn.commit()
    conn.close()


def create_session(
    user_id: int,
    username: str,
    role: str,
    employee_name: str,
    employee_id: str,
    department: str,
    remember_me: bool = False,
) -> str:
    _init_session_table()
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    expiry_minutes = 10080 if remember_me else 5

    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO user_sessions
            (session_id, user_id, username, role, employee_name, employee_id, department, last_active, remember_me, expiry_minutes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, user_id, username, role, employee_name, employee_id, department, now, int(remember_me), expiry_minutes),
    )
    conn.commit()
    conn.close()

    st.query_params["session"] = session_id
    st.session_state.session_id = session_id
    return session_id


def clear_session() -> None:
    session_id = st.session_state.get("session_id")
    if session_id:
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()

    st.query_params.pop("session", None)
    for key in [
        "authenticated",
        "user_role",
        "username",
        "user_id",
        "employee_name",
        "employee_id",
        "department",
        "session_id",
    ]:
        if key in st.session_state:
            del st.session_state[key]


def ensure_session(timeout_minutes: int | None = 5) -> bool:
    _init_session_table()
    now = datetime.now()
    session_id = st.session_state.get("session_id")
    if st.session_state.get("authenticated") and session_id:
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT last_active, expiry_minutes FROM user_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        if row is None:
            conn.close()
            clear_session()
            return False
        last_active = datetime.fromisoformat(row["last_active"])
        if timeout_minutes is not None and timeout_minutes > 0:
            expiry = timedelta(minutes=int(timeout_minutes))
            if now - last_active > expiry:
                conn.close()
                clear_session()
                return False
        cursor.execute(
            "UPDATE user_sessions SET last_active = ? WHERE session_id = ?",
            (now.isoformat(), session_id),
        )
        conn.commit()
        conn.close()
        return True

    if not st.session_state.get("authenticated"):
        session_id = st.query_params.get("session")
        if session_id:
            conn = db.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_id, username, role, employee_name, employee_id, department, last_active, expiry_minutes
                FROM user_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if row:
                last_active = datetime.fromisoformat(row["last_active"])
                if timeout_minutes is not None and timeout_minutes > 0:
                    expiry = timedelta(minutes=int(timeout_minutes))
                    if now - last_active > expiry:
                        cursor.execute(
                            "DELETE FROM user_sessions WHERE session_id = ?",
                            (session_id,),
                        )
                        conn.commit()
                        conn.close()
                        st.query_params.pop("session", None)
                        return False

                st.session_state.authenticated = True
                st.session_state.user_id = row["user_id"]
                st.session_state.username = row["username"]
                role = row["role"].lower() if isinstance(row["role"], str) else row["role"]
                st.session_state.user_role = role
                st.session_state.employee_name = row["employee_name"]
                st.session_state.employee_id = row["employee_id"]
                st.session_state.department = row["department"]
                st.session_state.session_id = session_id

                cursor.execute(
                    "UPDATE user_sessions SET last_active = ? WHERE session_id = ?",
                    (now.isoformat(), session_id),
                )
                conn.commit()
                conn.close()
                return True
            conn.close()

    return st.session_state.get("authenticated", False)

def require_roles(allowed_roles: tuple[str, ...]) -> None:
    """Stop page rendering unless user is authenticated and role is allowed."""
    if not st.session_state.get("authenticated"):
        st.switch_page("pages/0_Dashboard.py")
        st.stop()

    role = st.session_state.get("user_role")
    if role not in allowed_roles:
        st.error("🔒 Access denied for your role.")
        st.stop()

def apply_sidebar_style() -> None:
    """Hide Streamlit's default sidebar page list."""
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
                display: none !important;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: 1rem;
            }
            div[data-testid="stHorizontalBlock"] > div {
                flex: 1 1 220px;
                min-width: 200px;
            }
            section[data-testid="stSidebar"] .stButton > button {
                width: 100%;
                text-align: left;
                padding: 0.6rem 0.9rem;
                border-radius: 10px;
                border: 1px solid #6f7fe1;
                background: linear-gradient(90deg, #6b7be4 0%, #7a55b3 100%);
                color: #111111;
                font-weight: 600;
                transition: all 0.2s ease;
            }
            section[data-testid="stSidebar"] .stButton > button:hover {
                transform: translateX(4px);
                background: linear-gradient(90deg, #6474dd 0%, #6f4daa 100%);
                border-color: #6a78d7;
                color: #d32f2f;
            }
            section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background: linear-gradient(90deg, #6b7be4 0%, #7a55b3 100%);
                color: #111111;
                border: 1px solid #6f7fe1;
            }
            .page-header {
                text-align: center;
                padding: 1rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                margin: 0 0 2rem 0;
                font-weight: 700;
            }
            .page-header h1 {
                margin: 0;
                font-size: 2rem;
                letter-spacing: 0.3px;
            }
            .nav-user-badge {
                display: inline-block;
                padding: 0.2rem 0.6rem;
                border-radius: 999px;
                font-size: 0.75rem;
                font-weight: 700;
                margin-top: 0.5rem;
            }
            .nav-user-badge-admin {
                background: #764ba2;
                color: #fff;
            }
            .nav-user-badge-user {
                background: #4facfe;
                color: #fff;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str) -> None:
    """Render a consistent page header across pages."""
    apply_sidebar_style()
    st.markdown(
        f'<div class="page-header"><h1>{title}</h1></div>',
        unsafe_allow_html=True,
    )


def render_sidebar(active_label: Optional[str] = None) -> None:
    """Render the shared navigation sidebar."""
    apply_sidebar_style()

    authenticated = st.session_state.get("authenticated", False)
    role = st.session_state.get("user_role", "user")

    if not authenticated:
        role = "user"

    nav_items = [
        item for item in NAV_ITEMS if role in item["roles"]
    ]

    with st.sidebar:
        if authenticated:
            employee_name = st.session_state.get("employee_name") or "User"
            username = st.session_state.get("username") or "user"
            badge_class = (
                "nav-user-badge-admin" if role == "admin" else "nav-user-badge-user"
            )
            st.markdown(
                f"""
                <div style="text-align: center; padding: 0.5rem 0 1rem;">
                    <div style="width: 64px; height: 64px; background: #667eea; 
                        border-radius: 50%; margin: 0 auto 0.5rem; 
                        display: flex; align-items: center; justify-content: center; 
                        color: white; font-size: 1.5rem;">
                        {employee_name[0].upper()}
                    </div>
                    <div style="font-weight: 700;">{employee_name}</div>
                    <div style="opacity: 0.7; font-size: 0.85rem;">@{username}</div>
                    <span class="nav-user-badge {badge_class}">{role.upper()}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
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
            return

        st.markdown("---")
        st.markdown("### Navigation")

        for item in nav_items:
            button_type = "primary" if item["label"] == active_label else "secondary"
            if st.button(
                item["label"],
                use_container_width=True,
                key=f"sidebar_nav_{item['label']}",
                type=button_type,
                disabled=not authenticated and item["page"] != "pages/0_Dashboard.py",
            ):
                st.switch_page(item["page"])

        st.markdown("---")
        st.caption(f"Session time: {datetime.now().strftime('%H:%M')}")

        if authenticated:
            st.markdown("---")
            if st.button(
                "🚪 Logout",
                use_container_width=True,
                key="sidebar_logout",
                type="secondary",
            ):
                clear_session()
                st.switch_page("pages/0_Dashboard.py")
