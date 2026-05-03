# AfroxAI

Zanthoxylum bioactivity prediction pipeline using ECFP4 fingerprints and XGBoost with Tanimoto-based confidence scoring.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Artifacts

| File | Description |
|------|-------------|
| `artifacts/zanthoxylum_features.parquet` | Molecular feature matrix |
| `artifacts/zanthoxylum_ecfp4.npy` | ECFP4 fingerprint vectors |
| `artifacts/tanimoto_confidence_xgboost.json` | Trained model + confidence thresholds |
