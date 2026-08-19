"""
Implements the task classification module described in Section 5.2.

Rule-based classification of a natural-language analytics query into one of
the four benchmark task classes. Classification combines keyword signals
(single-word/phrase hits) with structural signals (patterns that indicate a
specific input shape -- an inline database schema, an inline CSV/numeric
block, a KPI value with an explicit benchmark comparison, or an audience/
sections specification).

Each matched signal contributes a weighted score to its task class. A match
closer to the start of the query counts more than one further in, so that
the primary instruction of a compound request (e.g. "pull X and write it up
for the board") determines the routing, consistent with the single-dispatch
design discussed in Section 7.4. If the highest-scoring class does not clear
FALLBACK_MIN_CONFIDENCE, classification falls back to FALLBACK_TASK_CLASS.
"""
import re
from dataclasses import dataclass, field

from talo.router import (
    SUPPORTED_TASK_CLASSES,
    FALLBACK_TASK_CLASS,
    FALLBACK_MIN_CONFIDENCE,
)

# A score total below this floor is treated as "no real evidence" even if
# one class nominally leads -- prevents a single weak, late keyword hit from
# producing spuriously high confidence.
MIN_TOTAL_SCORE = 1.5

# Label signals: (task_class, weight, compiled regex). Unambiguous, explicit
# field markers (an inline schema, or "Audience:"/"Sections:"/"KPI:" labels).
# These are NOT position-decayed: unlike narrative intent, a field label's
# position in a structured request reflects template layout, not priority
# (e.g. a report's "Audience:"/"Sections:" labels come after its data block,
# but still unambiguously mark the request as report generation).
_LABEL_SIGNALS: list[tuple[str, float, re.Pattern]] = [
    ("sql_generation", 5.0, re.compile(
        r"\bschema\s*:|\b\w+\s*\([^)]*\b(int|text|real|decimal|date|pk)\b",
        re.IGNORECASE,
    )),
    ("kpi_interpretation", 5.0, re.compile(r"\bkpi\s*:", re.IGNORECASE)),
    ("report_generation", 5.0, re.compile(
        r"\baudience\s*:|\bsections?\s*:", re.IGNORECASE,
    )),
]

# Structural signals: (task_class, weight, compiled regex). Narrative
# patterns that indicate intent -- position-decayed, so the earliest/primary
# instruction in a compound request dominates (see module docstring).
_STRUCTURAL_SIGNALS: list[tuple[str, float, re.Pattern]] = [
    # Explicit retrieval command.
    ("sql_generation", 3.0, re.compile(
        r"\bshow me\b|\bpull (the )?|\blist (the )?|\bget (the )?|"
        r"\bwhich\s+\w+(\s+\w+)?\s+have\b|\bhow many\b|\btop \d+\b|\btop (ten|five|three)\b",
        re.IGNORECASE,
    )),
    # Inline CSV/numeric block: header row with commas, or a run of >= 3
    # comma-separated numbers.
    ("anomaly_detection", 3.0, re.compile(
        r"(\d+(\.\d+)?\s*,\s*){2,}\d+(\.\d+)?|"
        r"^\s*[\w ]+(,[\w %]+){1,}\s*$",
        re.IGNORECASE | re.MULTILINE,
    )),
    # Single named metric with an explicit benchmark/target/prior comparison.
    ("kpi_interpretation", 3.0, re.compile(
        r"\bdown from\b|\bup from\b|\bcompared to\b|"
        r"\bagainst (target|benchmark)\b|\bvs\.?\s*(target|benchmark)\b|"
        r"\b(target|benchmark)\s*:|\bprior (period|month|quarter|year)\b",
        re.IGNORECASE,
    )),
    # Audience/stakeholder mention.
    ("report_generation", 3.0, re.compile(
        r"\bboard\b|\bleadership\b|\bstakeholders?\b|\bexecutives?\b|"
        r"\bc-suite\b|\bmanagement\b",
        re.IGNORECASE,
    )),
]

# Keyword signals: (task_class, weight, compiled regex over single words/phrases).
_KEYWORD_SIGNALS: list[tuple[str, float, re.Pattern]] = [
    ("sql_generation", 1.0, re.compile(
        r"\bselect\b|\bquery\b|\bsql\b|\bdatabase\b|\btable\b", re.IGNORECASE)),
    ("anomaly_detection", 1.0, re.compile(
        r"\banomal(y|ies)\b|\bunusual\w*\b|\boutliers?\b|\bspike\w*\b|"
        r"\bsuspicious\b|\bstands? out\b|\blook at\b|\binspect\b",
        re.IGNORECASE)),
    ("kpi_interpretation", 1.0, re.compile(
        r"\binterpret\w*\b|\bwhat does (that|this) mean\b|\btrend\b|\bwhy did\b|\bexplain\b",
        re.IGNORECASE)),
    ("report_generation", 1.0, re.compile(
        r"\breport\b|\bsummar(y|ise|ize)\b|\bwrite (it )?up\b|\boverview\b",
        re.IGNORECASE)),
]


@dataclass
class ClassificationResult:
    task_class: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    used_fallback: bool = False


def _position_weight(match_start: int, text_length: int) -> float:
    """Earlier matches count more. Ranges from 1.0 (start) down to 0.3 (end)."""
    if text_length <= 0:
        return 1.0
    normalized = match_start / text_length
    return 1.0 - normalized * 0.7


def _score_query(query: str) -> dict[str, float]:
    scores = {tc: 0.0 for tc in SUPPORTED_TASK_CLASSES}
    text_length = max(len(query), 1)

    for task_class, weight, pattern in _LABEL_SIGNALS:
        if pattern.search(query):
            scores[task_class] += weight

    for signals in (_STRUCTURAL_SIGNALS, _KEYWORD_SIGNALS):
        for task_class, weight, pattern in signals:
            match = pattern.search(query)
            if match:
                scores[task_class] += weight * _position_weight(match.start(), text_length)

    return scores


class TaskClassifier:
    """
    Rule-based classifier mapping a free-text analytics query to one of the
    four TALO task classes, with a confidence score and a fallback path for
    low-evidence input.
    """

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a natural-language analytics query.

        Args:
            query: The raw user query (as passed to run_talo.py or the
                Streamlit demo).

        Returns:
            ClassificationResult with the chosen task class, a confidence in
            [0, 1], the per-class signal scores, and whether the fallback
            path was used.
        """
        scores = _score_query(query or "")
        total = sum(scores.values())
        top_class = max(scores, key=scores.get)

        if total < MIN_TOTAL_SCORE:
            confidence = 0.0
        else:
            confidence = scores[top_class] / total

        if confidence < FALLBACK_MIN_CONFIDENCE:
            return ClassificationResult(
                task_class=FALLBACK_TASK_CLASS,
                confidence=confidence,
                scores=scores,
                used_fallback=True,
            )

        return ClassificationResult(
            task_class=top_class,
            confidence=confidence,
            scores=scores,
            used_fallback=False,
        )
