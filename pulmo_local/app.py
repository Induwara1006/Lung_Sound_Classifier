"""Streamlit UI for AeroVita Local."""

from __future__ import annotations

import sqlite3
import base64
from datetime import datetime, timezone
from pathlib import Path

import bcrypt
import plotly.express as px
import streamlit as st

import db
from config import DISCLAIMER, RECORDINGS_DIR
from inference import analyze_audio
from report import build_report


def _apply_theme() -> None:
    """Apply the local visual theme without changing application behavior."""
    lung_art = (
        "data:image/svg+xml;base64,"
        + base64.b64encode(
            b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 600">
            <defs><linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
            <stop stop-color="#0b2538"/><stop offset="1" stop-color="#167d83"/></linearGradient>
            <filter id="b"><feGaussianBlur stdDeviation="30"/></filter></defs>
            <rect width="900" height="600" fill="url(#g)"/>
            <circle cx="720" cy="90" r="180" fill="#6fe0d1" opacity=".16" filter="url(#b)"/>
            <path d="M450 110v330M450 180c-90-60-190-20-220 80-30 100 25 220 130 190 55-16 70-110 90-190M450 180c90-60 190-20 220 80 30 100-25 220-130 190-55-16-70-110-90-190" fill="none" stroke="#d7fff8" stroke-width="14" stroke-linecap="round" opacity=".7"/>
            <path d="M450 115c-24 35-30 70 0 105 30-35 24-70 0-105z" fill="#d7fff8" opacity=".8"/>
            </svg>"""
        ).decode("ascii")
    )
    st.markdown(
        f"""
        <style>
        :root {{ --ink:#102a43; --muted:#60758a; --teal:#087f82; --mint:#e8f7f4; --line:#dce8ee; }}
        .stApp {{ background-image:linear-gradient(rgba(247,251,252,.94),rgba(238,246,247,.96)),url('{lung_art}'); background-attachment:fixed; background-size:cover; color:var(--ink); font-family:Arial,sans-serif; }}
        [data-testid="stHeader"] {{ background:rgba(247,251,252,.86); }}
        [data-testid="stSidebar"] {{ background:linear-gradient(180deg,#0b2538 0%,#123e51 100%); }}
        [data-testid="stSidebar"] * {{ color:#e8f7f4 !important; }}
        [data-testid="stSidebar"] .stRadio label {{ padding:11px 12px; border-radius:10px; font-weight:600; }}
        [data-testid="stSidebar"] .stRadio label:hover {{ background:rgba(255,255,255,.1); }}
        h1,h2,h3 {{ font-family:Arial,sans-serif; color:var(--ink); letter-spacing:-.03em; }}
        .brand {{ padding:8px 0 24px; border-bottom:1px solid rgba(255,255,255,.16); margin-bottom:22px; }}
        .brand-mark {{ display:inline-flex; width:38px; height:38px; align-items:center; justify-content:center; border-radius:12px; background:#7be4d3; color:#0b2538; font-size:22px; margin-right:10px; }}
        .brand-name {{ font-family:Arial,sans-serif; font-size:20px; font-weight:800; vertical-align:middle; }}
        .eyebrow {{ color:var(--teal); font-weight:700; font-size:12px; letter-spacing:.12em; text-transform:uppercase; margin-bottom:6px; }}
        .hero {{ background-image:linear-gradient(90deg,rgba(8,41,57,.98) 0%,rgba(8,41,57,.78) 55%,rgba(8,41,57,.34) 100%),url('{lung_art}'); background-size:cover; background-position:center; border-radius:24px; padding:34px 38px; min-height:210px; color:white; margin:4px 0 26px; box-shadow:0 16px 35px rgba(16,42,67,.12); }}
        .hero h1 {{ color:white; margin:4px 0 10px; font-size:34px; }}
        .hero p {{ color:#d5eeeb; max-width:540px; font-size:16px; }}
        .panel {{ background:white; border:1px solid var(--line); border-radius:18px; padding:22px; box-shadow:0 8px 22px rgba(16,42,67,.05); margin-bottom:18px; }}
        .metric {{ background:white; border:1px solid var(--line); border-radius:16px; padding:18px 20px; box-shadow:0 8px 22px rgba(16,42,67,.04); }}
        .metric-label {{ color:var(--muted); font-size:13px; font-weight:600; }}
        .metric-value {{ color:var(--ink); font-family:Arial,sans-serif; font-size:28px; margin-top:5px; }}
        .notice {{ background:#e8f7f4; border-left:4px solid var(--teal); padding:14px 16px; border-radius:10px; color:#24515a; }}
        .stButton > button, .stDownloadButton > button {{ border-radius:10px; border:0; background:#087f82; color:white; font-weight:700; padding:10px 18px; }}
        .stButton > button:hover, .stDownloadButton > button:hover {{ background:#06666a; color:white; }}
        [data-testid="stFileUploader"] {{ background:white; border:1px dashed #7ab8b5; border-radius:16px; padding:8px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _page_header(label: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">{label}</div><h1>{title}</h1><p style="color:#60758a;margin-top:-12px;margin-bottom:24px">{subtitle}</p>', unsafe_allow_html=True)


def _session_doctor() -> dict[str, object] | None:
    return st.session_state.get("doctor")


def _show_auth() -> None:
    """Render login and registration tabs."""
    st.markdown('<div class="hero"><div class="eyebrow" style="color:#8debdc">LOCAL CLINICAL WORKSPACE</div><h1>Listen closer.<br>Decide with confidence.</h1><p>AeroVita helps clinicians organize lung sound screening securely on their own laptop.</p></div>', unsafe_allow_html=True)
    login, register = st.tabs(["Login", "Register"])
    with login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            doctor = db.get_doctor_by_email(email)
            if doctor and bcrypt.checkpw(password.encode(), doctor["password_hash"].encode()):
                st.session_state.doctor = {"id": doctor["id"], "name": doctor["name"]}
                st.rerun()
            else:
                st.error("Invalid email or password.")
    with register:
        with st.form("registration_form"):
            name = st.text_input("Name")
            email = st.text_input("Registration email")
            password = st.text_input("Registration password", type="password")
            confirmation = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account")
        if submitted:
            if not name.strip() or not email.strip() or len(password) < 8:
                st.error("Enter a name, email, and password with at least 8 characters.")
            elif password != confirmation:
                st.error("Passwords do not match.")
            else:
                password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
                try:
                    db.create_doctor(name, email, password_hash)
                except sqlite3.IntegrityError:
                    st.error("That email is already registered.")
                else:
                    st.success("Account created. You can now log in.")


def _patient_form(doctor_id: int) -> None:
    """Add a patient."""
    with st.form("patient_form"):
        fields = {
            "name": st.text_input("Full name"),
            "national_id": st.text_input("National ID"),
            "gender": st.selectbox("Gender", ["Female", "Male", "Other", "Prefer not to say"]),
            "birth_date": st.date_input("Birth date"),
            "hometown": st.text_input("Hometown"),
            "phone": st.text_input("Phone"),
            "notes": st.text_area("Notes"),
        }
        submitted = st.form_submit_button("Add patient")
    if submitted:
        required = ("name", "national_id", "hometown", "phone")
        if any(not str(fields[key]).strip() for key in required):
            st.error("Name, national ID, hometown, and phone are required.")
        else:
            fields["birth_date"] = str(fields["birth_date"])
            db.create_patient(doctor_id, fields)
            st.success("Patient added.")
            st.rerun()


def _patients_page(doctor_id: int) -> None:
    _page_header("CARE MANAGEMENT", "Patients", "Manage your patient records and keep every screening connected to the right person.")
    with st.expander("＋ Add a new patient", expanded=False):
        _patient_form(doctor_id)
    search = st.text_input("Search by name, national ID, or phone")
    patients = db.list_patients(doctor_id, search)
    for patient in patients:
        with st.expander(f"{patient['name']} — {patient['national_id']}"):
            st.write(
                f"Gender: {patient['gender']} | Birth date: {patient['birth_date']} | "
                f"Phone: {patient['phone']}"
            )
            st.write(f"Hometown: {patient['hometown']}")
            st.write(patient["notes"] or "No notes.")
            if st.button("Delete patient", key=f"delete_{patient['id']}"):
                db.delete_patient(patient["id"], doctor_id)
                st.rerun()


def _patient_select(doctor_id: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    patients = [dict(row) for row in db.list_patients(doctor_id)]
    if not patients:
        st.info("Add a patient before uploading a recording.")
        return {}, []
    labels = {f"{p['name']} ({p['national_id']})": p for p in patients}
    selected = st.selectbox("Patient", list(labels))
    return labels[selected], patients


def _display_result(result: dict[str, object], chart_key: str = "latest") -> None:
    if result["is_healthy"]:
        st.success(f"Healthy — confidence {float(result['healthy_score']):.2%}")
    else:
        top = result.get("top_predictions", [])
        st.warning("Abnormal screening result")
        if top:
            chart_data = [{"Disease": label, "Probability": probability} for label, probability in top]
            st.dataframe(chart_data, use_container_width=True, hide_index=True)
            left, right = st.columns(2)
            with left:
                st.plotly_chart(
                    px.pie(chart_data, names="Disease", values="Probability"),
                    use_container_width=True,
                    key=f"{chart_key}_pie",
                )
            with right:
                st.plotly_chart(
                    px.bar(chart_data, x="Disease", y="Probability"),
                    use_container_width=True,
                    key=f"{chart_key}_bar",
                )
    st.caption(DISCLAIMER)


def _upload_page(doctor_id: int) -> None:
    _page_header("SCREENING WORKSPACE", "Upload & Analyze", "Upload a clean WAV recording to generate a local decision-support result.")
    patient, _ = _patient_select(doctor_id)
    if not patient:
        return
    upload = st.file_uploader("Upload a WAV file", type=["wav"])
    if upload is None:
        return
    if st.button("Analyze recording"):
        patient_dir = RECORDINGS_DIR / str(patient["id"])
        patient_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = patient_dir / f"{timestamp}.wav"
        destination.write_bytes(upload.getbuffer())
        try:
            with st.spinner("Preprocessing audio and running analysis..."):
                result = analyze_audio(destination)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            destination.unlink(missing_ok=True)
            st.error(str(exc))
            return
        recording_id = db.create_recording(patient["id"], str(destination))
        db.create_prediction(recording_id, result)
        st.session_state.latest_result = result
        st.session_state.latest_patient = patient
        st.session_state.latest_recording_date = timestamp
    if st.session_state.get("latest_result") and st.session_state.get("latest_patient", {}).get("id") == patient["id"]:
        _display_result(st.session_state.latest_result)
        report_bytes = build_report(
            str(st.session_state.doctor["name"]),
            st.session_state.latest_patient,
            str(st.session_state.latest_recording_date),
            st.session_state.latest_result,
        )
        st.download_button("Download PDF report", report_bytes, "aerovita-report.pdf", "application/pdf")


def _history_page(doctor_id: int) -> None:
    _page_header("CLINICAL TIMELINE", "History", "Review previous recordings, predictions, and downloadable reports.")
    patient, _ = _patient_select(doctor_id)
    if not patient:
        return
    history = db.list_patient_history(int(patient["id"]), doctor_id)
    if not history:
        st.info("No recordings for this patient.")
        return
    for row in history:
        with st.expander(str(row["uploaded_at"])):
            st.write(f"File: {row['file_path']}")
            if row["is_healthy"] is None:
                st.write("Not analyzed")
            else:
                prediction = {
                    "is_healthy": bool(row["is_healthy"]),
                    "healthy_score": row["healthy_score"],
                    "top_predictions": [
                        (row[key], row[f"{key.replace('_label', '_prob')}"])
                        for key in ("top1_label", "top2_label", "top3_label")
                        if row[key] is not None
                    ],
                }
                _display_result(prediction, chart_key=f"history_{row['id']}")
                report = build_report(str(st.session_state.doctor["name"]), patient, row["uploaded_at"], prediction)
                st.download_button("Download report", report, f"report-{row['id']}.pdf", "application/pdf")


def main() -> None:
    """Run the Streamlit application."""
    db.init_db()
    st.set_page_config(page_title="AeroVita Local", page_icon="🫁", layout="wide")
    _apply_theme()
    if _session_doctor() is None:
        _show_auth()
        return
    doctor = _session_doctor()
    st.sidebar.markdown('<div class="brand"><span class="brand-mark">◌</span><span class="brand-name">AeroVita</span><div style="font-size:11px;color:#9bd8d1;margin:8px 0 0 50px">LOCAL LUNG SOUND INTELLIGENCE</div></div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div style="font-size:12px;color:#9bd8d1;margin-bottom:8px">SIGNED IN AS</div><div style="font-weight:700;margin-bottom:20px">Dr. {doctor["name"]}</div>', unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", ["Dashboard", "Patients", "Upload & Analyze", "History", "Logout"])
    if page == "Logout":
        st.session_state.clear()
        st.rerun()
    if page == "Dashboard":
        st.markdown('<div class="hero"><div class="eyebrow" style="color:#8debdc">GOOD TO SEE YOU</div><h1>Your clinical workspace,<br>in one calm view.</h1><p>Review patients, analyze recordings, and keep your screening history organized — entirely on this laptop.</p></div>', unsafe_allow_html=True)
        patients = db.list_patients(int(doctor["id"]))
        st.markdown(f'<div class="metric"><div class="metric-label">PATIENTS IN YOUR CARE</div><div class="metric-value">{len(patients)}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="notice" style="margin-top:22px"><strong>Screening reminder</strong><br>{DISCLAIMER}</div>', unsafe_allow_html=True)
    elif page == "Patients":
        _patients_page(int(doctor["id"]))
    elif page == "Upload & Analyze":
        _upload_page(int(doctor["id"]))
    else:
        _history_page(int(doctor["id"]))


if __name__ == "__main__":
    main()
