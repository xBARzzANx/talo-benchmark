"""
Implements the SQL extraction and anomaly parsing variants documented in
Section 4.4.5 of the thesis (parser/extractor sensitivity analysis).

These functions are ports of the logic that was originally inline in
evaluate_phase1.py, evaluate_phase2.py, and diagnose_scoring.py. They are
byte-for-byte behavioral copies -- extracting them here does not change any
previously reported score.

SQL extractors:
    extract_sql_fence_only     -- Phase 1 (reported)
    extract_sql_keyword_line   -- Phase 2 (reported), Phase 3 (primary)
    extract_sql_terminal_block -- sensitivity analysis only

Anomaly parsers:
    parse_anomaly_permissive -- Phase 1/2 (reported), Phase 3 (primary)
    parse_anomaly_strict     -- sensitivity analysis only
"""
import json
import re


def extract_sql_fence_only(content: str) -> str:
    """Strip only ```sql code fences. Used in Phase 1 (reported)."""
    return re.sub(r"```sql|```", "", content or "").strip()


def extract_sql_keyword_line(content: str) -> str:
    """
    Fence removal, then -- if a SELECT is present -- keep only the lines
    from the first SQL keyword (SELECT/WITH/INSERT) onward. Handles
    Chain-of-Thought outputs that prepend reasoning text before the query.
    Used in Phase 2 (reported) and as the primary variant for Phase 3.
    """
    predicted = extract_sql_fence_only(content)
    if "SELECT" in predicted.upper():
        sql_lines, in_sql = [], False
        for line in predicted.split("\n"):
            if any(kw in line.upper() for kw in ["SELECT", "WITH", "INSERT"]):
                in_sql = True
            if in_sql:
                sql_lines.append(line)
        predicted = "\n".join(sql_lines).strip()
    return predicted


def extract_sql_terminal_block(content: str) -> str:
    """
    Take the last fenced code block if one exists, else the text from the
    last SELECT/WITH keyword onward; truncate at the first semicolon.
    Used only for the extractor sensitivity analysis (Section 4.4.5).
    """
    if not content:
        return ""
    text = content.strip()

    fences = re.findall(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fences:
        candidate = fences[-1].strip()
    else:
        matches = list(re.finditer(r"\b(SELECT|WITH)\b", text, re.IGNORECASE))
        if not matches:
            return text
        candidate = text[matches[-1].start():].strip()

    if ";" in candidate:
        candidate = candidate.split(";")[0].strip() + ";"
    return candidate


def parse_anomaly_permissive(model_output: str, n_rows: int) -> list[int]:
    """
    Parse a JSON list if present, else fall back to a regex over all
    integers found anywhere in the text that fall within [0, n_rows).
    Used in Phase 1/2 (reported) and as the primary variant for Phase 3.
    """
    try:
        parsed = json.loads(model_output)
        if isinstance(parsed, list):
            return [int(i) for i in parsed if 0 <= int(i) < n_rows]
    except (json.JSONDecodeError, ValueError):
        pass
    numbers = re.findall(r"\b(\d+)\b", model_output)
    return [int(n) for n in numbers if 0 <= int(n) < n_rows]


def parse_anomaly_strict(model_output: str, n_rows: int) -> list[int]:
    """
    Parse only an explicit JSON list: the whole output, else the last
    bracketed group in the text. No free-integer fallback.
    Used only for the anomaly parser sensitivity analysis (Section 4.4.5).
    """
    if not model_output:
        return []
    try:
        parsed = json.loads(model_output.strip())
        if isinstance(parsed, list):
            return sorted({int(i) for i in parsed if 0 <= int(i) < n_rows})
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    cleaned = re.sub(r"```json|```", "", model_output)
    blocks = re.findall(r"\[[^\[\]]*\]", cleaned)
    for block in reversed(blocks):
        try:
            parsed = json.loads(block)
            if isinstance(parsed, list):
                return sorted({int(i) for i in parsed
                               if isinstance(i, (int, float)) and 0 <= int(i) < n_rows})
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return []
