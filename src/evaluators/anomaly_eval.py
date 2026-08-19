"""F1-based evaluator for anomaly detection."""
from sklearn.metrics import precision_score, recall_score, f1_score
from src.evaluators.extractors import parse_anomaly_permissive as parse_anomaly_indices


def evaluate_anomaly(model_output: str, ground_truth_indices: list[int], n_rows: int) -> dict:
    predicted = set(parse_anomaly_indices(model_output, n_rows))
    ground_truth = set(ground_truth_indices)
    y_true = [1 if i in ground_truth else 0 for i in range(n_rows)]
    y_pred = [1 if i in predicted else 0 for i in range(n_rows)]
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 3),
        "predicted_indices": sorted(predicted),
    }
