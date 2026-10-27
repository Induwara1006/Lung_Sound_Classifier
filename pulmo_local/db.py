"""SQLite persistence helpers for AeroVita Local."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import DATABASE_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: Path | str = DATABASE_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with foreign keys enabled."""
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db(db_path: Path | str = DATABASE_PATH) -> None:
    """Create all application tables if they do not exist."""
    with get_connection(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_id INTEGER NOT NULL REFERENCES doctors(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                national_id TEXT NOT NULL,
                gender TEXT NOT NULL,
                birth_date TEXT NOT NULL,
                hometown TEXT NOT NULL,
                phone TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
                file_path TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id INTEGER NOT NULL UNIQUE REFERENCES recordings(id) ON DELETE CASCADE,
                is_healthy INTEGER NOT NULL,
                healthy_score REAL NOT NULL,
                top1_label TEXT,
                top1_prob REAL,
                top2_label TEXT,
                top2_prob REAL,
                top3_label TEXT,
                top3_prob REAL,
                created_at TEXT NOT NULL
            );
            """
        )


def create_doctor(name: str, email: str, password_hash: str, db_path: Path | str = DATABASE_PATH) -> int:
    """Create a doctor and return the new ID."""
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO doctors (name, email, password_hash) VALUES (?, ?, ?)",
            (name.strip(), email.strip().lower(), password_hash),
        )
        return int(cursor.lastrowid)


def get_doctor_by_email(email: str, db_path: Path | str = DATABASE_PATH) -> sqlite3.Row | None:
    """Return a doctor by case-insensitive email."""
    with get_connection(db_path) as connection:
        return connection.execute(
            "SELECT id, name, email, password_hash FROM doctors WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()


def create_patient(doctor_id: int, data: dict[str, str], db_path: Path | str = DATABASE_PATH) -> int:
    """Create a patient belonging to a doctor."""
    values = (
        doctor_id,
        data["name"].strip(),
        data["national_id"].strip(),
        data["gender"].strip(),
        data["birth_date"].strip(),
        data["hometown"].strip(),
        data["phone"].strip(),
        data.get("notes", "").strip(),
        _now(),
    )
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """INSERT INTO patients
            (doctor_id, name, national_id, gender, birth_date, hometown, phone, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values,
        )
        return int(cursor.lastrowid)


def list_patients(doctor_id: int, search: str = "", db_path: Path | str = DATABASE_PATH) -> list[sqlite3.Row]:
    """List only a doctor's patients, optionally matching common fields."""
    pattern = f"%{search.strip()}%"
    with get_connection(db_path) as connection:
        return connection.execute(
            """SELECT * FROM patients
            WHERE doctor_id = ? AND (name LIKE ? OR national_id LIKE ? OR phone LIKE ?)
            ORDER BY name""",
            (doctor_id, pattern, pattern, pattern),
        ).fetchall()


def get_patient(patient_id: int, doctor_id: int, db_path: Path | str = DATABASE_PATH) -> sqlite3.Row | None:
    """Get a patient only when it belongs to the doctor."""
    with get_connection(db_path) as connection:
        return connection.execute(
            "SELECT * FROM patients WHERE id = ? AND doctor_id = ?",
            (patient_id, doctor_id),
        ).fetchone()


def delete_patient(patient_id: int, doctor_id: int, db_path: Path | str = DATABASE_PATH) -> bool:
    """Delete a doctor-owned patient and cascaded records."""
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM patients WHERE id = ? AND doctor_id = ?",
            (patient_id, doctor_id),
        )
        return cursor.rowcount == 1


def create_recording(patient_id: int, file_path: str, db_path: Path | str = DATABASE_PATH) -> int:
    """Store an uploaded recording path."""
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            "INSERT INTO recordings (patient_id, file_path, uploaded_at) VALUES (?, ?, ?)",
            (patient_id, file_path, _now()),
        )
        return int(cursor.lastrowid)


def create_prediction(recording_id: int, result: dict[str, Any], db_path: Path | str = DATABASE_PATH) -> int:
    """Store an inference result."""
    top = result.get("top_predictions", [])
    values = (
        recording_id,
        int(result["is_healthy"]),
        float(result["healthy_score"]),
        top[0][0] if len(top) > 0 else None,
        float(top[0][1]) if len(top) > 0 else None,
        top[1][0] if len(top) > 1 else None,
        float(top[1][1]) if len(top) > 1 else None,
        top[2][0] if len(top) > 2 else None,
        float(top[2][1]) if len(top) > 2 else None,
        _now(),
    )
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """INSERT INTO predictions
            (recording_id, is_healthy, healthy_score, top1_label, top1_prob,
             top2_label, top2_prob, top3_label, top3_prob, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values,
        )
        return int(cursor.lastrowid)


def list_patient_history(patient_id: int, doctor_id: int, db_path: Path | str = DATABASE_PATH) -> list[sqlite3.Row]:
    """List recordings and predictions for a doctor-owned patient."""
    with get_connection(db_path) as connection:
        return connection.execute(
            """SELECT r.*, p.* FROM recordings r
            JOIN patients pt ON pt.id = r.patient_id AND pt.doctor_id = ?
            LEFT JOIN predictions p ON p.recording_id = r.id
            WHERE r.patient_id = ? ORDER BY r.uploaded_at DESC""",
            (doctor_id, patient_id),
        ).fetchall()
