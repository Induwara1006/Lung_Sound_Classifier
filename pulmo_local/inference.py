"""Audio validation, preprocessing, feature extraction, and model inference."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Protocol

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from config import (
    BINARY_MODEL_PATH,
    CLASS_NAMES,
    CNN_MODEL_PATH,
    FIXED_LENGTH,
    HEALTHY_IS_ONE,
    HEALTHY_THRESHOLD,
    HOP_LENGTH,
    HYBRID_MODEL_PATH,
    MIN_AUDIO_SECONDS,
    N_FFT,
    N_MFCC,
    SAMPLE_RATE,
    SEGMENT_SECONDS,
    SILENCE_PEAK_THRESHOLD,
)


class PredictiveModel(Protocol):
    """Minimal protocol implemented by Keras models."""

    input_shape: Any
    output_shape: Any

    def predict(self, inputs: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Predict outputs for a batch."""


def validate_wav(path: Path | str) -> None:
    """Reject unsupported, short, or essentially silent WAV files."""
    file_path = Path(path)
    if file_path.suffix.lower() != ".wav":
        raise ValueError("Only WAV audio files are supported.")
    try:
        info = sf.info(str(file_path))
        if info.duration < MIN_AUDIO_SECONDS:
            raise ValueError(f"Audio must be at least {MIN_AUDIO_SECONDS} seconds long.")
        data, _ = sf.read(str(file_path), always_2d=False, dtype="float32")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read the WAV file: {exc}") from exc
    if data.size == 0 or float(np.max(np.abs(data))) < SILENCE_PEAK_THRESHOLD:
        raise ValueError(
            f"Audio is essentially silent; peak amplitude must be at least {SILENCE_PEAK_THRESHOLD}."
        )


def normalize_waveform(y: np.ndarray) -> np.ndarray:
    """Apply the exact training min-max normalization to [-1, 1]."""
    minimum = np.min(y)
    maximum = np.max(y)
    denominator = maximum - minimum
    if denominator == 0:
        return np.zeros_like(y, dtype=np.float32)
    normalized = 2 * ((y - minimum) / denominator) - 1
    return normalized.astype(np.float32)


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract the trained MFCC, mel, and chroma feature tensor."""
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mfcc = librosa.util.fix_length(mfcc, size=FIXED_LENGTH, axis=1)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = librosa.util.fix_length(mel, size=FIXED_LENGTH, axis=1)
    mel = np.resize(mel, (N_MFCC, FIXED_LENGTH))
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    chroma = librosa.util.fix_length(chroma, size=FIXED_LENGTH, axis=1)
    chroma = np.resize(chroma, (N_MFCC, FIXED_LENGTH))
    return np.stack((mfcc, mel, chroma), axis=-1).astype(np.float32)


def prepare_segments(path: Path | str, temporary_directory: Path) -> list[Path]:
    """Convert audio and export six-second stereo WAV segments."""
    audio = AudioSegment.from_wav(str(path))
    audio = audio.set_channels(2).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
    segment_length_ms = SEGMENT_SECONDS * 1000
    segment_paths: list[Path] = []
    for index, start in enumerate(range(0, len(audio), segment_length_ms)):
        segment = audio[start : start + segment_length_ms]
        if len(segment) < segment_length_ms:
            segment += AudioSegment.silent(duration=segment_length_ms - len(segment))
        output_path = temporary_directory / f"segment_{index}.wav"
        segment.export(str(output_path), format="wav")
        segment_paths.append(output_path)
    if not segment_paths:
        raise ValueError("The WAV file did not contain any audio segments.")
    return segment_paths


def load_segment_features(segment_paths: list[Path]) -> np.ndarray:
    """Load, normalize, and extract features from each exported segment."""
    features: list[np.ndarray] = []
    for segment_path in segment_paths:
        y, sr = librosa.load(str(segment_path), sr=None)
        features.append(extract_features(normalize_waveform(y), sr))
    return np.asarray(features, dtype=np.float32)


def load_models() -> tuple[PredictiveModel, PredictiveModel, PredictiveModel]:
    """Load binary, CNN, and hybrid models from the configured workspace paths."""
    missing = [
        path.name
        for path in (BINARY_MODEL_PATH, CNN_MODEL_PATH, HYBRID_MODEL_PATH)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s) in the workspace root: " + ", ".join(missing)
        )
    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Install requirements.txt before loading models."
        ) from exc
    return (
        load_model(BINARY_MODEL_PATH),
        load_model(CNN_MODEL_PATH),
        load_model(HYBRID_MODEL_PATH),
    )


def cached_load_models() -> tuple[PredictiveModel, PredictiveModel, PredictiveModel]:
    """Load models once per Streamlit process."""
    try:
        import streamlit as st
    except ImportError:
        return load_models()
    return st.cache_resource(load_models)()


def _binary_score(output: np.ndarray) -> float:
    """Convert a Keras sigmoid output to a scalar score."""
    return float(np.asarray(output).reshape(-1)[0])


def analyze_audio(
    path: Path | str,
    models: tuple[PredictiveModel, PredictiveModel, PredictiveModel] | None = None,
) -> dict[str, Any]:
    """Run the complete binary-gated analysis pipeline for a WAV file."""
    file_path = Path(path)
    validate_wav(file_path)
    with tempfile.TemporaryDirectory(prefix="pulmosense_") as temporary_name:
        segment_paths = prepare_segments(file_path, Path(temporary_name))
        features = load_segment_features(segment_paths)
        binary_model, cnn_model, hybrid_model = models or cached_load_models()
        binary_outputs = np.asarray(binary_model.predict(features, verbose=0))
        binary_scores = np.asarray(binary_outputs).reshape(-1)
        healthy_score = float(np.mean(binary_scores))
        healthy = healthy_score >= HEALTHY_THRESHOLD if HEALTHY_IS_ONE else healthy_score < HEALTHY_THRESHOLD
        result: dict[str, Any] = {
            "is_healthy": healthy,
            "healthy_score": healthy_score,
            "binary_outputs": binary_scores.tolist(),
            "segment_count": len(segment_paths),
            "top_predictions": [],
        }
        if healthy:
            return result
        cnn_outputs = np.asarray(cnn_model.predict(features, verbose=0))
        hybrid_outputs = np.asarray(hybrid_model.predict(features, verbose=0))
        ensemble_segments = 0.4 * cnn_outputs + 0.6 * hybrid_outputs
        probabilities = np.mean(ensemble_segments, axis=0)
        order = np.argsort(probabilities)[::-1][:3]
        result["ensemble_probabilities"] = probabilities.tolist()
        result["top_predictions"] = [
            (CLASS_NAMES[int(index)], float(probabilities[index])) for index in order
        ]
        result["cnn_outputs"] = cnn_outputs.tolist()
        result["hybrid_outputs"] = hybrid_outputs.tolist()
        return result
