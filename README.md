# PET/CT-based pCR Prediction Calculator

This Streamlit application implements the final seven-predictor Random Forest model for estimating pathological complete response (pCR) after neoadjuvant chemoimmunotherapy in resectable NSCLC.

## Predictors

- Pre-treatment SUVmax (SUVmax1)
- Post-treatment SUVmax (SUVmax2)
- SUV reduction ratio = `(SUVmax1 - SUVmax2) / SUVmax1`
- Smoking status
- RECIST response: Responder (CR/PR) or non-responder (SD/PD)
- Histological subtype: LUAD or LUSC
- Clinical N stage: cN0-1 or cN2

## Validation and threshold

The primary internal validation was training-only repeated nested stratified five-fold cross-validation repeated five times. The validation cohort was not used for feature selection, hyperparameter tuning, or model selection. The displayed decision threshold is 0.426, derived from aggregated outer-fold out-of-fold predictions using the Youden index. Bootstrap-OOB was retained as a supplementary robustness analysis.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app is usually available at `http://localhost:8501`.

## Streamlit Community Cloud

Upload the contents of this folder to the root of a GitHub repository, set `app.py` as the main file, and use Python 3.12.

## Intended use

Research use only. The calculator does not replace pathological assessment, multidisciplinary review, or clinical judgment. Prospective and geographically diverse external validation is required before clinical implementation.
