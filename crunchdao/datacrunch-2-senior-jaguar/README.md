# CrunchDAO DataCrunch-2 — senior-jaguar

## Overview
- **Competition**: `datacrunch-2` (Crowdsourced Asset Pricing Tournament)
- **Model**: `senior-jaguar` (`huge-roy/senior-jaguar`)
- **Format**: `UNSTRUCTURED` / `OFFLINE`
- **Active Round**: **Round 41** (October 11, 2026 – October 18, 2026, 12:00 UTC)
- **Target Metric**: **Spearman Rank Correlation** grouped by `moon` (temporal market eras)
- **Model Architecture**: Purged Rank Ensemble (`LightGBMRegressor` + `Ridge` L2 Regularization) with uniform percentile ranking.

---

## Workspace Structure
```
crunchdao/datacrunch-2-senior-jaguar/
├── .crunchdao/
│   └── project.json         # Competition & project metadata
├── resources/
│   └── model.joblib         # Serialized ensemble weights
├── data/                    # Tournament parquet datasets (x_train, y_train, x_test)
├── prediction/              # Generated predictions (prediction.parquet)
├── main.py                  # Entrypoint with train() and infer() hooks
├── requirements.txt         # Pinned runner dependencies
└── README.md
```

---

## Local Testing
To verify the model pipeline end-to-end:
```bash
py main.py
```
This runs a synthetic simulation of the CrunchDAO offline scoring engine with 6 moons, computes rank-normalized predictions, and verifies the Spearman correlation score.

---

## Token Renewal & CLI Submission
CrunchDAO temporary setup clone tokens are valid for ~15–30 minutes upon creation on the Hub. When ready to push or renew credentials:

1. Open your model page on CrunchDAO Hub:
   👉 **[https://hub.crunchdao.com/competitions/datacrunch-2/projects/14415/senior-jaguar](https://hub.crunchdao.com/competitions/datacrunch-2/projects/14415/senior-jaguar)**
2. Click **Submit** or **Copy Setup Command** to get a fresh clone token.
3. Update the project token:
   ```bash
   py -m crunch update-token <NEW_TOKEN>
   ```
4. Push and submit the model:
   ```bash
   py -m crunch push -m "Senior Jaguar: LightGBM + Ridge rank ensemble for Round 41"
   ```
