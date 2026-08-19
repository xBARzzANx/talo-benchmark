"""SQL execution accuracy evaluator."""
import sqlite3


def normalize_result(rows: list) -> frozenset:
    return frozenset(
        frozenset(str(round(v, 2)) if isinstance(v, float) else str(v) for v in row)
        for row in rows
    )


def execute_sql(query: str, db_path: str) -> tuple:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return rows, None
    except Exception as e:
        return None, str(e)


def evaluate_sql(predicted_sql: str, ground_truth_sql: str, db_path: str) -> dict:
    pred_rows, pred_err = execute_sql(predicted_sql, db_path)
    if pred_err:
        return {"score": 0, "execution_error": pred_err, "match": False}
    gt_rows, gt_err = execute_sql(ground_truth_sql, db_path)
    if gt_err:
        return {"score": 0, "execution_error": f"Ground truth error: {gt_err}", "match": False}
    match = normalize_result(pred_rows) == normalize_result(gt_rows)
    return {"score": int(match), "execution_error": None, "match": match}
