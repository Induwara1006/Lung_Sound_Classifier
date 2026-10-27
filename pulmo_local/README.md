# PulmoSense Local

PulmoSense Local is a local-only Streamlit desktop demo for lung sound screening. It stores doctors, patients, recordings, and predictions in SQLite and does not send audio or patient data to cloud services.

## Setup

Use Python 3.10 or newer. From this directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`pydub` requires ffmpeg. Install ffmpeg separately and ensure its executable is on `PATH` (for example, with `winget install Gyan.FFmpeg`).

## Models

This project currently resolves the three required `.h5` model files from the workspace root:

- `lung_sound_binary_classification_model.h5`
- `lung_sound_multiclass_classification_model.h5`
- `lung_sound_Hybrid_multiclass_classification_model.h5`

The optional `ensemble_model.h5` is not used for the app's primary inference because the requested behavior combines the CNN and CNN-LSTM models with weights 0.4 and 0.6. For a packaged deployment, model paths can be moved into the `models/` directory and `config.py` can be updated.

## Run

From `pulmo_local`:

```powershell
streamlit run app.py
```

The SQLite database is created automatically as `pulmo_local.sqlite3`. Uploaded WAV files are saved under `recordings/<patient_id>/`.

## Disclaimer

This tool is for screening and decision support only and is not a medical diagnosis. Predictions for rare conditions are unreliable.
