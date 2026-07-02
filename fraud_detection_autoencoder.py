"""
fraud_detection_autoencoder.py

Fraud Detection using PyOD's AutoEncoder (unsupervised deep learning anomaly
detection) on the anonymized credit card transactions dataset from Kaggle:
https://www.kaggle.com/datasets/whenamancodes/fraud-detection

The AutoEncoder learns to reconstruct "normal" transactions. Fraudulent
transactions, being rare and structurally different, tend to produce a
higher reconstruction error, which PyOD uses as the anomaly/outlier score.

Author: <Your Name>
Course/Assignment: Deep Learning for Anomaly Detection - Fraud Detection Assignment

Usage:
    python fraud_detection_autoencoder.py --data creditcard.csv

If --data is not supplied, or the file cannot be found, the script falls
back to generating a small synthetic dataset that mirrors the real
dataset's schema (Time, V1-V28, Amount, Class) purely so the pipeline can
be demonstrated end-to-end without the actual Kaggle file. For the
assignment submission, download the real 'creditcard.csv' from the Kaggle
link above and pass it via --data.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend so this runs headless too
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
)

from pyod.models.auto_encoder import AutoEncoder

# --------------------------------------------------------------------------
# Logging setup - best practice: use logging instead of scattered print()
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42  # fixed seed for reproducibility


def load_dataset(data_path: str | None) -> pd.DataFrame:
    """
    Load the Kaggle credit card fraud dataset.

    Expected columns: Time, V1..V28 (PCA-anonymized features), Amount, Class
    (Class == 1 -> fraud, Class == 0 -> normal transaction).

    If the file is missing, a synthetic dataset with the same schema is
    generated so the rest of the pipeline can still be demonstrated.

    Args:
        data_path: path to creditcard.csv, or None.

    Returns:
        DataFrame with the transaction data.
    """
    if data_path and Path(data_path).exists():
        logger.info("Loading real dataset from %s", data_path)
        df = pd.read_csv(data_path)
        return df

    logger.warning(
        "Dataset file not found at %s. Generating a synthetic dataset "
        "with the same schema (Time, V1-V28, Amount, Class) for "
        "demonstration purposes. Download the real file from "
        "https://www.kaggle.com/datasets/whenamancodes/fraud-detection "
        "and pass it with --data creditcard.csv for the real assignment run.",
        data_path,
    )
    return _generate_synthetic_dataset()


def _generate_synthetic_dataset(
    n_normal: int = 20000, n_fraud: int = 100
) -> pd.DataFrame:
    """
    Create a synthetic stand-in dataset that mimics the structure and
    class imbalance of the real Kaggle credit card fraud dataset:
    28 PCA-like features (V1-V28), a Time column, an Amount column,
    and a highly imbalanced binary Class label.

    Fraud rows are drawn from a shifted/scaled distribution so the
    AutoEncoder has a genuine (if simplified) pattern to separate.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    n_features = 28
    normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))
    fraud = rng.normal(loc=2.5, scale=1.8, size=(n_fraud, n_features))

    time_normal = rng.integers(0, 172792, size=n_normal)
    time_fraud = rng.integers(0, 172792, size=n_fraud)

    amount_normal = rng.exponential(scale=60, size=n_normal)
    amount_fraud = rng.exponential(scale=250, size=n_fraud)

    cols = [f"V{i}" for i in range(1, n_features + 1)]
    df_normal = pd.DataFrame(normal, columns=cols)
    df_normal["Time"] = time_normal
    df_normal["Amount"] = amount_normal
    df_normal["Class"] = 0

    df_fraud = pd.DataFrame(fraud, columns=cols)
    df_fraud["Time"] = time_fraud
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = 1

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    ordered_cols = ["Time"] + cols + ["Amount", "Class"]
    return df[ordered_cols]


def preprocess(df: pd.DataFrame):
    """
    Split features/labels, scale numeric features, and produce a
    train/test split. Because AutoEncoder-based fraud detection is
    typically framed as unsupervised/semi-supervised, we train the
    model mainly on the feature space but keep labels aside purely
    for evaluation (this is standard practice with PyOD).

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    logger.info("Preprocessing data: scaling Amount/Time and splitting train/test")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Scale Amount and Time; V1-V28 are already PCA components (roughly
    # standardized already in the real dataset, but we scale everything
    # uniformly here for consistency and best practice).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Train size: %d (fraud=%d) | Test size: %d (fraud=%d)",
        len(y_train),
        int(y_train.sum()),
        len(y_test),
        int(y_test.sum()),
    )

    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), scaler


def build_and_train_autoencoder(X_train: np.ndarray, contamination: float) -> AutoEncoder:
    """
    Build and train a PyOD AutoEncoder for unsupervised anomaly detection.

    Args:
        X_train: scaled training feature matrix.
        contamination: expected proportion of outliers (fraud) in the data;
            used by PyOD to set the internal decision threshold.

    Returns:
        A fitted AutoEncoder model.
    """
    n_features = X_train.shape[1]

    # Symmetric encoder/decoder architecture: compress down to a small
    # bottleneck, then reconstruct. Deeper/wider layers can be used for
    # larger real-world datasets.
    hidden_neurons = [
        min(64, n_features),
        16,
        16,
        min(64, n_features),
    ]

    logger.info("Building AutoEncoder with hidden layers: %s", hidden_neurons)

    model = AutoEncoder(
        hidden_neuron_list=hidden_neurons,
        contamination=contamination,
        epoch_num=30,
        batch_size=64,
        lr=1e-3,
        random_state=RANDOM_STATE,
    )

    logger.info("Training AutoEncoder ...")
    model.fit(X_train)
    logger.info("Training complete.")

    return model


def evaluate_model(model: AutoEncoder, X_test: np.ndarray, y_test: np.ndarray, output_dir: Path):
    """
    Evaluate the trained AutoEncoder on the held-out test set and produce:
      - ROC-AUC and Average Precision (PR-AUC) scores
      - classification report / confusion matrix
      - ROC curve plot and reconstruction-error histogram saved as PNGs

    Args:
        model: fitted PyOD AutoEncoder.
        X_test: scaled test feature matrix.
        y_test: true labels (0 = normal, 1 = fraud).
        output_dir: directory to save plots to.
    """
    logger.info("Scoring test set and computing metrics ...")

    # decision_function returns the raw anomaly/outlier score
    # (reconstruction error); predict() returns binary labels using the
    # contamination-derived threshold.
    y_scores = model.decision_function(X_test)
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)

    logger.info("ROC-AUC: %.4f", roc_auc)
    logger.info("Average Precision (PR-AUC): %.4f", pr_auc)

    report = classification_report(y_test, y_pred, target_names=["Normal", "Fraud"])
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("FRAUD DETECTION - AUTOENCODER EVALUATION RESULTS")
    print("=" * 60)
    print(f"ROC-AUC Score:            {roc_auc:.4f}")
    print(f"Average Precision (PR-AUC): {pr_auc:.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print("=" * 60 + "\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- ROC Curve plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_scores, ax=ax, name="AutoEncoder")
    ax.set_title("ROC Curve - Fraud Detection AutoEncoder")
    fig.tight_layout()
    roc_path = output_dir / "roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    # --- Reconstruction error histogram (normal vs fraud) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(y_scores[y_test == 0], bins=50, alpha=0.6, label="Normal", density=True)
    ax.hist(y_scores[y_test == 1], bins=50, alpha=0.6, label="Fraud", density=True)
    ax.set_xlabel("Reconstruction Error (Anomaly Score)")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error Distribution: Normal vs. Fraud")
    ax.legend()
    fig.tight_layout()
    hist_path = output_dir / "reconstruction_error_hist.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    logger.info("Saved plots to: %s and %s", roc_path, hist_path)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a PyOD AutoEncoder for credit card fraud detection."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="creditcard.csv",
        help="Path to the Kaggle creditcard.csv dataset "
             "(https://www.kaggle.com/datasets/whenamancodes/fraud-detection).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output plots/screenshots to.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df = load_dataset(args.data)
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Fraud cases: %d (%.4f%% of data)",
                int(df["Class"].sum()), 100 * df["Class"].mean())

    X_train, X_test, y_train, y_test, _scaler = preprocess(df)

    # contamination = expected fraud rate in the training data;
    # PyOD uses this to set the anomaly-score threshold for predict().
    contamination = max(min(y_train.mean(), 0.5), 0.001)

    model = build_and_train_autoencoder(X_train, contamination)

    evaluate_model(model, X_test, y_test, Path(args.output_dir))


if __name__ == "__main__":
    sys.exit(main())
