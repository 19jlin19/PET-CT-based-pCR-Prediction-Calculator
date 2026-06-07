# PET/CT-based pCR Prediction Calculator

This folder contains an online prediction calculator for the final Random Forest model used to predict pathological complete response (pCR) after neoadjuvant therapy in NSCLC.

## Files

- `app.py`: Streamlit web application.
- `predictor_core.py`: Core prediction functions.
- `model/final_randomforest_pipeline.joblib`: Serialized sklearn pipeline.
- `model/metadata.json`: Model metadata, feature schema, threshold, and performance summary.
- `sample_input.csv`: Example batch prediction file.
- `requirements.txt`: Python dependencies.

## Required predictors

| Variable | Type | Allowed values / notes |
|---|---|---|
| SUVmax1 | Continuous | Pre-treatment SUVmax |
| SUVmax2 | Continuous | Post-treatment SUVmax |
| SUV_ratio | Continuous | `(SUVmax1 - SUVmax2) / SUVmax1`; can be auto-calculated |
| Smoking | Binary | `0` = No, `1` = Yes |
| RECIST_response | Categorical | `Responder` or `Non-responder` |
| Histological | Categorical | `LUAD` or `LUSC` |
| cN | Categorical | `cN0-1` or `cN2` |

## Run locally

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open in your browser, usually at:

```text
http://localhost:8501
```

## Deploy online

This app can be deployed to Streamlit Community Cloud, a hospital intranet server, or any Python-compatible cloud server.

For Streamlit Cloud:

1. Upload this folder to a private GitHub repository.
2. Set `app.py` as the main file.
3. Ensure `requirements.txt` is present.
4. Deploy.

## Model note

The decision threshold is 0.398, derived from aggregated bootstrap-OOB internal validation. The tool is intended for research use only and should not replace clinical judgment.
