# Synthetic data generator for employees and attendance
import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
from faker import Faker

from navigation import apply_sidebar_style, ensure_session, render_page_header, render_sidebar, require_roles


st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_sidebar_style()
ensure_session(timeout_minutes=None)
render_sidebar("🧪 Synthetic Data")
require_roles(("admin",))

render_page_header("🧪 Synthetic Data Generator")

st.markdown("Generate synthetic employee and attendance data for model training.")
st.caption(
    "Algorithms used: Faker (en_NG) for names, jobs, emails, phones; randomized lists/ranges for departments and dates. "
    "Outliers use extreme times (0–4h, 22–23h) and placeholder location/IP values. "
    "Missing check-outs convert a portion of check-outs to check-ins."
)

faker = Faker("en_NG")
departments = ["HR", "IT", "Finance", "Sales", "Operations", "Marketing", "Support"]


def _random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def generate_employees(count: int) -> pd.DataFrame:
    rows = []
    used_names = set()
    for i in range(count):
        emp_id = str(i + 1)
        name = faker.unique.name()
        if name in used_names:
            name = f"{name} {i + 1}"
        used_names.add(name)
        dept = random.choice(departments)
        title = faker.job()
        hire_date = _random_date(datetime.now() - timedelta(days=365 * 5), datetime.now()).date()
        rows.append(
            {
                "employee_id": emp_id,
                "employee_name": name,
                "department": dept,
                "job_title": title,
                "hire_date": hire_date.isoformat(),
                "email": faker.company_email(),
                "phone": faker.phone_number(),
                "photo_path": "",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_active": 1,
            }
        )
    return pd.DataFrame(rows)


def generate_attendance(
    employees_df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    outlier_rate: float,
    missing_checkout_rate: float,
    extra_records_rate: float,
) -> pd.DataFrame:
    if employees_df.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "employee_id",
                "employee_name",
                "department",
                "job_title",
                "date",
                "time",
                "check_type",
                "is_anomaly_true",
                "ip_address",
                "location_city",
                "location_region",
                "location_country",
                "latitude",
                "longitude",
                "isp",
                "created_at",
            ]
        )

    rows = []
    row_id = 1

    # Precompute employee-day keys so anomaly rates apply across the dataset
    employee_days = []
    day = start_date
    while day <= end_date:
        if day.weekday() < 5:
            for _, emp in employees_df.iterrows():
                employee_days.append((str(emp["employee_id"]), day.date().isoformat()))
        day += timedelta(days=1)

    employee_ids = [str(emp_id) for emp_id in employees_df["employee_id"].tolist()]

    def _sample_keys(rate: float) -> set:
        if not employee_days or rate <= 0:
            return set()
        # Limit anomalies to a subset of employees to avoid "everyone has anomalies"
        emp_k = max(1, int(round(len(employee_ids) * rate)))
        affected_emps = set(random.sample(employee_ids, emp_k))
        eligible_days = [key for key in employee_days if key[0] in affected_emps]
        if not eligible_days:
            return set()
        k = max(1, int(round(len(eligible_days) * rate)))
        return set(random.sample(eligible_days, min(k, len(eligible_days))))

    outlier_keys = _sample_keys(outlier_rate)
    missing_keys = _sample_keys(missing_checkout_rate)
    extra_keys = _sample_keys(extra_records_rate)

    day = start_date
    while day <= end_date:
        if day.weekday() < 5:
            for _, emp in employees_df.iterrows():
                emp_id = str(emp["employee_id"])
                key = (emp_id, day.date().isoformat())
                is_outlier = key in outlier_keys
                missing_checkout = key in missing_keys
                has_extra_records = key in extra_keys
                def _time_with_jitter(center_hour: int) -> tuple[int, int]:
                    jitter_minutes = random.randint(-30, 30)
                    total_minutes = center_hour * 60 + jitter_minutes
                    total_minutes = max(0, min(23 * 60 + 59, total_minutes))
                    return total_minutes // 60, total_minutes % 60

                def _outlier_time() -> tuple[int, int]:
                    hour = random.choice([0, 1, 2, 3, 4, 22, 23])
                    minute = random.randint(0, 59)
                    return hour, minute

                if not is_outlier:
                    check_in_hour, check_in_min = _time_with_jitter(8)
                    check_out_hour, check_out_min = _time_with_jitter(17)
                else:
                    # Outliers are intentionally far from expected hours
                    check_in_hour, check_in_min = _outlier_time()
                    check_out_hour, check_out_min = _outlier_time()

                def _add_record(check_type: str, hour: int, minute: int, is_anomaly: bool) -> None:
                    nonlocal row_id
                    time_value = (
                        datetime.combine(day.date(), datetime.min.time())
                        + timedelta(hours=hour, minutes=minute)
                    ).time()
                    rows.append(
                        {
                            "id": row_id,
                            "employee_id": emp["employee_id"],
                            "employee_name": emp["employee_name"],
                            "department": emp.get("department", ""),
                            "job_title": emp.get("job_title", ""),
                            "date": day.date().isoformat(),
                            "time": time_value.strftime("%H:%M:%S"),
                            "check_type": check_type,
                            "is_anomaly_true": int(is_anomaly),
                            "ip_address": "" if not is_outlier else "0.0.0.0",
                            "location_city": "Abuja",
                            "location_region": "FCT",
                            "location_country": "Nigeria",
                            "latitude": "9.0765",
                            "longitude": "7.3986",
                            "isp": "NG ISP",
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                    row_id += 1

                # Only records with out-of-scope times are labeled as anomalies.
                if is_outlier:
                    _add_record("check_in", check_in_hour, check_in_min, True)
                    if not missing_checkout:
                        _add_record("check_out", check_out_hour, check_out_min, True)
                else:
                    if missing_checkout:
                        # Missing checkout: mark the single check-in as anomalous with out-of-scope time
                        out_h, out_m = _outlier_time()
                        _add_record("check_in", out_h, out_m, True)
                    else:
                        _add_record("check_in", check_in_hour, check_in_min, False)
                        _add_record("check_out", check_out_hour, check_out_min, False)

                if has_extra_records:
                    # Extra records are anomalous; use out-of-scope times so flags align with timing
                    out_h1, out_m1 = _outlier_time()
                    out_h2, out_m2 = _outlier_time()
                    _add_record("check_out", out_h1, out_m1, True)
                    _add_record("check_in", out_h2, out_m2, True)
        day += timedelta(days=1)
    return pd.DataFrame(rows)


with st.form("synthetic_generator"):
    st.subheader("Generator Settings")
    st.info("Configure employee count, date range, and anomaly injection rates.")
    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        employee_count = st.number_input("Employees to generate", min_value=1, max_value=500, value=50, step=1)
    with cfg2:
        start_date = st.date_input("Start date", value=datetime(2025, 11, 1))
    with cfg3:
        end_date = st.date_input("End date", value=datetime(2026, 1, 24))

    col1, col2, col3 = st.columns(3)
    with col1:
        outlier_rate = st.slider("Outlier rate (%)", min_value=0.0, max_value=5.0, value=2.0, step=0.2)
    with col2:
        missing_checkout_rate = st.slider(
            "Missing check-outs (%)",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.2,
        )
    with col3:
        extra_records_rate = st.slider(
            "Extra records (%)",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.2,
        )

    include_anomaly_label = st.checkbox(
        "Include `is_anomaly_true` column in attendance table/export",
        value=True,
        help="Turn off to generate clean attendance data without anomaly labels in the output.",
    )

    generate_btn = st.form_submit_button("Generate Data", use_container_width=True)

if "synthetic_employees" not in st.session_state:
    st.session_state.synthetic_employees = pd.DataFrame()
if "synthetic_attendance" not in st.session_state:
    st.session_state.synthetic_attendance = pd.DataFrame()

if generate_btn:
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time())
    if end_dt < start_dt:
        st.warning("End date is earlier than start date. Using start date as end date.")
        end_dt = start_dt

    st.session_state.synthetic_employees = generate_employees(int(employee_count))
    st.session_state.synthetic_attendance = generate_attendance(
        st.session_state.synthetic_employees,
        start_dt,
        end_dt,
        outlier_rate / 100.0,
        missing_checkout_rate / 100.0,
        extra_records_rate / 100.0,
    )

if not st.session_state.synthetic_employees.empty:
    st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
    st.subheader("Generated Employees")
    st.dataframe(st.session_state.synthetic_employees, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download Employees CSV",
        data=st.session_state.synthetic_employees.to_csv(index=False),
        file_name="synthetic_employees.csv",
        mime="text/csv",
        use_container_width=True,
    )

if not st.session_state.synthetic_attendance.empty:
    st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
    st.subheader("Generated Attendance")
    attendance_view = st.session_state.synthetic_attendance.copy()
    if not include_anomaly_label:
        attendance_view = attendance_view.drop(columns=["is_anomaly_true"], errors="ignore")
    st.dataframe(attendance_view, use_container_width=True, hide_index=True)
    st.download_button(
        label="Download Attendance CSV",
        data=attendance_view.to_csv(index=False),
        file_name="synthetic_attendance.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Footer
st.markdown('<hr style="border: none; border-top: 2px solid #6078ea; margin: 10px 0;">', unsafe_allow_html=True)
footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
with footer_col2:
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>💼 <strong>AI-Powered Attendance System</strong></p>
        <p style="font-size: 0.9em;">Synthetic Data Module</p>
        <p style="font-size: 0.8em;">Python Faker, Random List, NumPy, Pandas</p>
        <p style="font-size: 0.8em;">Version 2.1 | Developed by: Itoro Udonyah (NOU234244897) | <a href="https://github.com/itoroudonyah" target="_blank">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)
