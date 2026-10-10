# Model Card: PET/CT-based pCR Random Forest Predictor

## Intended use
Research-oriented individualized estimation of pathological complete response after neoadjuvant chemoimmunotherapy in resectable non-small cell lung cancer.

## Inputs
SUVmax1, SUVmax2, SUV reduction ratio, smoking status, RECIST response, histological subtype, and clinical N stage.

## Output
Estimated pCR probability and a threshold-based classification using the aggregated outer-fold out-of-fold Youden threshold of 0.426.

## Reported performance
- Training apparent AUC: 0.948
- Training-only repeated nested-CV AUC: 0.823 (95% CI 0.802–0.844)
- Validation AUC: 0.874 (95% CI 0.768–0.958)
- Validation accuracy: 0.824
- Validation sensitivity: 0.667
- Validation specificity: 0.909
- Validation Brier score: 0.141
- Supplementary aggregated Bootstrap-OOB AUC: 0.830

## Limitations
The model was developed from retrospective data with 203 training and 51 validation patients. The validation cohort is relatively small. Prospective, geographically diverse external validation is required before routine clinical use. The tool is intended for research use only and should not replace pathological assessment or clinical judgment.
