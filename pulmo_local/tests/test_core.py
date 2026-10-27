"""Focused tests for persistence and deterministic inference helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import bcrypt
import pytest

PROJECT_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import db  # noqa: E402


def test_doctor_and_patient_are_scoped(tmp_path: Path) -> None:
    database = tmp_path / "test.sqlite3"
    db.init_db(database)
    first = db.create_doctor("A", "a@example.com", "hash", database)
    second = db.create_doctor("B", "b@example.com", "hash", database)
    patient_id = db.create_patient(
        first,
        {
            "name": "Patient",
            "national_id": "N1",
            "gender": "Other",
            "birth_date": "2000-01-01",
            "hometown": "Town",
            "phone": "123",
            "notes": "",
        },
        database,
    )
    assert db.get_patient(patient_id, first, database) is not None
    assert db.get_patient(patient_id, second, database) is None
    assert db.delete_patient(patient_id, second, database) is False


def test_password_hash_round_trip() -> None:
    password = b"correct horse battery staple"
    hashed = bcrypt.hashpw(password, bcrypt.gensalt())
    assert bcrypt.checkpw(password, hashed)
    assert not bcrypt.checkpw(b"wrong", hashed)


def test_normalization_and_feature_shape() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("librosa")
    import inference

    assert np.array_equal(inference.normalize_waveform(np.ones(4)), np.zeros(4))
    normalized = inference.normalize_waveform(np.array([-2.0, 0.0, 2.0]))
    assert np.allclose(normalized, [-1.0, 0.0, 1.0])
    features = inference.extract_features(np.zeros(44100 * 3, dtype=np.float32), 44100)
    assert features.shape == (128, 264, 3)
