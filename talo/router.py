"""
TALO Router -- Task-Aware LLM Optimizer
Implements the routing component described in Section 5.3 of the thesis.

The Alignment Matrix (model + prompt strategy per task class) is not
hardcoded here. It is read from configs/alignment_matrix.yaml, the single
source of truth derived from the Phase 1 and Phase 2 benchmark results.

Scores reflect the evaluation protocol defined in Sections 4.4.1 and 4.4.2.
Parser and extractor sensitivity is documented in Section 4.4.5.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

ALIGNMENT_MATRIX_PATH = Path(__file__).resolve().parent.parent / "configs" / "alignment_matrix.yaml"


@dataclass
class RoutingDecision:
    task_class: str
    model_id: str
    strategy: str
    confidence: str  # "high" | "medium" | "low"
    rationale: str


def _load_alignment_matrix_config(path: Path = ALIGNMENT_MATRIX_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_CONFIG = _load_alignment_matrix_config()

# Alignment Matrix -- derived from Phase 1 + Phase 2 benchmark results.
# Source of truth: configs/alignment_matrix.yaml
ALIGNMENT_MATRIX: dict = _CONFIG["routing"]

SUPPORTED_TASK_CLASSES = list(ALIGNMENT_MATRIX.keys())

# Fallback used by talo/classifier.py when task classification confidence
# is insufficient. See the "fallback" block in configs/alignment_matrix.yaml.
FALLBACK_TASK_CLASS: str = _CONFIG["fallback"]["task_class"]
FALLBACK_MIN_CONFIDENCE: float = _CONFIG["fallback"]["min_confidence"]


class TALORouter:
    """
    Rule-based router that maps task classes to optimal model + strategy pairs.
    Routing decisions are derived empirically from Phase 1 and Phase 2 benchmark
    results. This implementation serves as the core routing component of the
    TALO framework.
    """

    def __init__(self):
        self.alignment_matrix = ALIGNMENT_MATRIX

    def route(self, task_class: str) -> RoutingDecision:
        """
        Route a task class to the optimal model and prompt strategy.

        Args:
            task_class: One of the four supported task classes.

        Returns:
            RoutingDecision with model_id, strategy, confidence, and rationale.

        Raises:
            ValueError: If task_class is not supported.
        """
        if task_class not in self.alignment_matrix:
            raise ValueError(
                f"Unsupported task class: '{task_class}'. "
                f"Supported classes: {SUPPORTED_TASK_CLASSES}"
            )

        entry = self.alignment_matrix[task_class]
        return RoutingDecision(
            task_class=task_class,
            model_id=entry["model_id"],
            strategy=entry["strategy"],
            confidence=entry["confidence"],
            rationale=entry["rationale"],
        )

    def route_all(self) -> dict[str, RoutingDecision]:
        """Return routing decisions for all supported task classes."""
        return {tc: self.route(tc) for tc in SUPPORTED_TASK_CLASSES}

    def get_alignment_matrix(self) -> dict:
        """Return the full alignment matrix with scores and metadata."""
        return self.alignment_matrix

    def summary(self) -> str:
        """Return a human-readable summary of all routing decisions."""
        lines = ["TALO Routing Summary", "=" * 50]
        for tc, entry in self.alignment_matrix.items():
            lines.append(f"\nTask Class:  {tc}")
            lines.append(f"  Model:     {entry['model_id']}")
            lines.append(f"  Strategy:  {entry['strategy']}")
            lines.append(f"  Score P1:  {entry['phase1_score']} ({entry['metric']})")
            lines.append(f"  Score P2:  {entry['phase2_score']} ({entry['strategy']})")
            lines.append(f"  Confidence:{entry['confidence']}")
        return "\n".join(lines)


if __name__ == "__main__":
    router = TALORouter()
    print(router.summary())
    print("\n--- Single Route Example ---")
    decision = router.route("sql_generation")
    print(f"Task:     {decision.task_class}")
    print(f"Model:    {decision.model_id}")
    print(f"Strategy: {decision.strategy}")
    print(f"Confidence: {decision.confidence}")
