"""Application configuration for AeroVita Local."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
MODEL_DIR = WORKSPACE_DIR
RECORDINGS_DIR = PROJECT_DIR / "recordings"
DATABASE_PATH = PROJECT_DIR / "pulmo_local.sqlite3"

BINARY_MODEL_PATH = MODEL_DIR / "lung_sound_binary_classification_model.h5"
CNN_MODEL_PATH = MODEL_DIR / "lung_sound_multiclass_classification_model.h5"
HYBRID_MODEL_PATH = MODEL_DIR / "lung_sound_Hybrid_multiclass_classification_model.h5"
SAVED_ENSEMBLE_MODEL_PATH = MODEL_DIR / "ensemble_model.h5"

SAMPLE_RATE = 44100
SEGMENT_SECONDS = 6
MIN_AUDIO_SECONDS = 3
SILENCE_PEAK_THRESHOLD = 0.01
N_FFT = 2048
HOP_LENGTH = 512
N_MFCC = 128
FIXED_LENGTH = 264
HEALTHY_THRESHOLD = 0.5
HEALTHY_IS_ONE = True

# TODO: Verify this index order against the training notebook before clinical use.
CLASS_NAMES = [
    "Asthma",
    "Bronchiectasis",
    "Bronchiolitis",
    "Bronchitis",
    "COPD",
    "Lung Fibrosis",
    "Pleural Effusion",
    "Pneumonia",
    "URTI",
]

DISCLAIMER = (
    "This tool is for screening and decision support only and is not a medical "
    "diagnosis. Predictions for rare conditions are unreliable."
)
