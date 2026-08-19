"""Unit tests for task evaluators."""
import pytest
from src.evaluators.anomaly_eval import parse_anomaly_indices, evaluate_anomaly
from src.evaluators.sql_eval import normalize_result


def test_anomaly_parse_json():
    assert parse_anomaly_indices("[3, 7]", n_rows=10) == [3, 7]

def test_anomaly_parse_text_fallback():
    assert parse_anomaly_indices("Anomalies at rows 3 and 7.", n_rows=10) == [3, 7]

def test_anomaly_eval_perfect():
    result = evaluate_anomaly("[3]", ground_truth_indices=[3], n_rows=5)
    assert result["f1"] == 1.0

def test_anomaly_eval_miss():
    result = evaluate_anomaly("[]", ground_truth_indices=[3], n_rows=5)
    assert result["f1"] == 0.0

def test_sql_normalize_order_independence():
    rows_a = [(1, "Alice"), (2, "Bob")]
    rows_b = [(2, "Bob"), (1, "Alice")]
    assert normalize_result(rows_a) == normalize_result(rows_b)
