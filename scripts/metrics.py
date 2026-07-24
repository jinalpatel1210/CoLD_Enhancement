import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

def classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }

def full_report(y_true, y_pred):
    return classification_report(y_true, y_pred, zero_division=0)

def noise_detection_metrics(keep_mask, y_noisy, y_clean):
    truly_noisy = y_noisy != y_clean
    removed = ~keep_mask
    tp = int(np.sum(removed & truly_noisy))
    fp = int(np.sum(removed & ~truly_noisy))
    fn = int(np.sum(~removed & truly_noisy))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    kept = keep_mask
    residual = float(np.mean(truly_noisy[kept])) if kept.sum() else 0.0
    return {
        "detector_precision": precision,
        "detector_recall": recall,
        "detector_f1": f1,
        "input_noise_rate": float(np.mean(truly_noisy)),
        "residual_noise_rate": residual,
        "kept_fraction": float(np.mean(keep_mask)),
    }
