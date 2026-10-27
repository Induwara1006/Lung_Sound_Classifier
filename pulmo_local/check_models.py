"""Verify model shapes and run inference for a WAV path."""

from __future__ import annotations

import argparse
from pathlib import Path

from inference import analyze_audio, cached_load_models


def main() -> int:
    """Run model and pipeline diagnostics."""
    parser = argparse.ArgumentParser(description="Check AeroVita model files.")
    parser.add_argument("wav_path", type=Path)
    args = parser.parse_args()
    models = cached_load_models()
    names = ("binary", "cnn", "hybrid")
    for name, model in zip(names, models):
        print(f"{name}: input_shape={model.input_shape}, output_shape={model.output_shape}")
    result = analyze_audio(args.wav_path, models=models)
    print("Raw binary outputs:", result["binary_outputs"])
    if result["is_healthy"]:
        print("Healthy score:", result["healthy_score"])
    else:
        print("CNN segment outputs:", result["cnn_outputs"])
        print("Hybrid segment outputs:", result["hybrid_outputs"])
        print("Ensemble probabilities:", result["ensemble_probabilities"])
        print("Top predictions:", result["top_predictions"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
