# Fraud Detection using PyOD AutoEncoder

Unsupervised deep-learning fraud detection on anonymized credit card
transactions, using PyOD's `AutoEncoder` outlier detector.

## Overview

This project trains an AutoEncoder neural network on transaction data.
The AutoEncoder learns to compress and reconstruct "normal" transactions.
Because fraudulent transactions are rare and statistically different from
normal ones, the model reconstructs them poorly — producing a **higher
reconstruction error**. PyOD uses this reconstruction error as the
anomaly/outlier score to flag likely fraud.

## Dataset

**Source:** [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/whenamancodes/fraud-detection)

The dataset contains anonymized European credit card transactions from
September 2013. It has 284,807 transactions, of which only 492 (~0.17%)
are fraudulent — a highly imbalanced dataset, which is typical of
real-world fraud detection problems.

Columns:

- `Time` – seconds elapsed since the first transaction
- `V1`–`V28` – anonymized features from a PCA transformation (the original
  features could not be published due to confidentiality)
- `Amount` – transaction amount
- `Class` – target label (`1` = fraud, `0` = normal)

**To run this project with the real data:**

1. Download `creditcard.csv` from the Kaggle link above (Kaggle account required).
2. Place it in this project folder.
3. Run: `python fraud_detection_autoencoder.py --data creditcard.csv`

> **Note on the included run:** My sandboxed execution environment for
> generating this deliverable does not have network access to Kaggle, so
> the results/screenshots included in this submission were produced using
> a small **synthetic dataset that mirrors the real schema** (same
> columns, same extreme class imbalance). The code is written to load and
> run identically against the real `creditcard.csv` — simply pass the
> real file with `--data` and re-run. Metrics will differ (the real
> dataset is larger and the fraud pattern is more subtle), but the
> pipeline, architecture, and evaluation approach are unchanged.

## Project Structure

```
fraud_detection/
├── fraud_detection_autoencoder.py   # Main source code
├── requirements.txt                 # Manifest / dependency file
├── README.md                        # This file
└── output/
    ├── roc_curve.png                     # ROC curve plot
    ├── reconstruction_error_hist.png     # Reconstruction error distribution
    └── console_output_screenshot.png     # Terminal output screenshot
```

## Environment Setup

1. Install Python 3.9+ (https://www.python.org/downloads/)
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install dependencies from the manifest file:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python fraud_detection_autoencoder.py --data creditcard.csv --output-dir output
```

Arguments:

- `--data` — path to the Kaggle `creditcard.csv` file (defaults to
  `creditcard.csv` in the current directory; falls back to a synthetic
  dataset with a warning if not found)
- `--output-dir` — directory to save the ROC curve / histogram plots to
  (default: `output/`)

## Model Details

- **Algorithm:** PyOD `AutoEncoder` (`pyod.models.auto_encoder.AutoEncoder`)
- **Architecture:** symmetric encoder/decoder, hidden layers `[64, 16, 16, 64]`
  (bottleneck of 16 neurons), fitted with 30 epochs, batch size 64, Adam
  optimizer (lr=1e-3)
- **Preprocessing:** `StandardScaler` applied to all features (`Time`,
  `V1`-`V28`, `Amount`)
- **Train/test split:** 70/30, stratified on the `Class` label so both
  splits preserve the fraud ratio
- **Contamination parameter:** set to the empirical fraud rate of the
  training data, which PyOD uses to derive the anomaly-score threshold
  for binary predictions
- **Evaluation metrics:** ROC-AUC, Average Precision (PR-AUC), confusion
  matrix, and a full classification report (precision/recall/F1 for both
  classes) — PR-AUC and recall on the fraud class matter most here since
  accuracy is a misleading metric on such an imbalanced dataset

## Sample Results (synthetic demonstration run)

| Metric                     | Value  |
| -------------------------- | ------ |
| ROC-AUC                    | 1.0000 |
| PR-AUC (Average Precision) | 0.9978 |
| Fraud recall               | 1.00   |
| Fraud precision            | 0.94   |

See `output/console_output_screenshot.png` for the full console output
and `output/roc_curve.png` / `output/reconstruction_error_hist.png` for
the evaluation plots. On the real, larger, and noisier Kaggle dataset,
expect somewhat lower (but still strong) scores — a well-tuned AutoEncoder
typically achieves ROC-AUC in the 0.93–0.97 range on this specific
dataset, in line with published benchmarks.

## References

- PyOD documentation: https://pyod.readthedocs.io/en/latest/pyod.models.html#module-pyod.models.auto_encoder
- PyOD GitHub: https://github.com/yzhao062/pyod
- Dataset: https://www.kaggle.com/datasets/whenamancodes/fraud-detection
- Reference notebook: https://github.com/PacktPublishing/Deep-Learning-and-XAI-Techniques-for-Anomaly-Detection/blob/main/Chapter1/PyOD_autoencoder/chapter1_pyod_autoencoder.ipynb
