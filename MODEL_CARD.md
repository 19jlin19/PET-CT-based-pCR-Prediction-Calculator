# Model Card: PET/CT-based pCR Random Forest Predictor

## Intended use
Research-oriented individualized prediction of pathological complete response after neoadjuvant therapy in non-small cell lung cancer.

## Inputs
SUVmax1, SUVmax2, SUV ratio, Smoking, RECIST response, Histology, and cN status.

## Output
Predicted probability of pCR and binary classification using the bootstrap-OOB-derived threshold of 0.398.

## Reported performance
- Training AUC: 0.943
- Aggregated bootstrap-OOB AUC: 0.830
- Mean bootstrap-OOB AUC: 0.828
- Validation AUC: 0.874, 95% CI 0.766-0.962
- Validation Brier score: 0.141

## Limitations
The model was developed from retrospective data and requires prospective and external validation before routine clinical use.
