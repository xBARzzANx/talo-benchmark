"""
Implements the end-to-end orchestration pipeline described in Section 5.5.

Wires together task classification (talo.classifier), routing
(talo.router), prompt selection (talo.prompt_selector), and the model
client (src.models.model_client) into a single call: input -> classify ->
route -> build prompt -> call -> validate -> structured result.

Classification uncertainty and prompt/API errors are captured on the
returned TALOResult rather than raised, so a caller (run_talo.py, app.py)
can always render a result.
"""
from dataclasses import dataclass
from typing import Optional

from talo.classifier import TaskClassifier, ClassificationResult
from talo.router import TALORouter, RoutingDecision
from talo.prompt_selector import PromptSelector, PromptSelectionError
from src.models.model_client import call_model

# Placeholder slot values used to fill template slots that a raw free-text
# query cannot supply (e.g. an SQL schema, or KPI benchmark figures). Real
# task input (as in the benchmark dataset) should be passed via **slot_values
# to TALOOrchestrator.run() to override these.
_DEFAULT_SLOTS = {
    "sql_generation": {"schema": "(not provided)"},
    "anomaly_detection": {},
    "kpi_interpretation": {
        "kpi_name": "(unspecified)",
        "value": "(unspecified)",
        "benchmark": "(unspecified)",
        "prior_period": "(unspecified)",
    },
    "report_generation": {
        "audience": "(unspecified)",
        "sections": "(unspecified)",
    },
}

# Which template slot the raw query text fills, per task class.
_QUERY_SLOT = {
    "sql_generation": "question",
    "anomaly_detection": "data",
    "kpi_interpretation": "context",
    "report_generation": "data",
}


@dataclass
class TALOResult:
    query: str
    classification: ClassificationResult
    routing: Optional[RoutingDecision] = None
    prompt: Optional[str] = None
    output: Optional[str] = None
    dry_run: bool = True
    error: Optional[str] = None


class TALOOrchestrator:
    """
    End-to-end TALO pipeline: query -> classification -> routing ->
    prompt -> (optional) model call -> structured result.
    """

    def __init__(self):
        self.classifier = TaskClassifier()
        self.router = TALORouter()
        self.prompt_selector = PromptSelector()

    def run(self, query: str, dry_run: bool = True, **slot_values) -> TALOResult:
        """
        Run the full TALO pipeline for a single query.

        Args:
            query: Raw natural-language analytics query.
            dry_run: If True (default), classification/routing/prompt are
                computed but no model call is made -- output stays None.
                Only set to False with explicit user confirmation that a
                paid API call may be made (see CLAUDE.md, Regel 2).
            **slot_values: Explicit template slot values (e.g. schema/data/
                kpi_name), used in place of the placeholder defaults for the
                resolved task class. Useful when the caller has structured
                task input rather than only a free-text query.

        Returns:
            A TALOResult. Classification always succeeds (falls back to
            FALLBACK_TASK_CLASS on low confidence). A routing or prompt
            error is captured on `result.error`; the pipeline never raises.
        """
        classification = self.classifier.classify(query)
        task_class = classification.task_class

        try:
            routing = self.router.route(task_class)
        except ValueError as e:
            return TALOResult(
                query=query, classification=classification,
                dry_run=dry_run, error=str(e),
            )

        slots = {
            _QUERY_SLOT[task_class]: query,
            **_DEFAULT_SLOTS[task_class],
            **slot_values,
        }

        try:
            prompt = self.prompt_selector.build_prompt(task_class, routing.strategy, **slots)
        except PromptSelectionError as e:
            return TALOResult(
                query=query, classification=classification, routing=routing,
                dry_run=dry_run, error=str(e),
            )

        if dry_run:
            return TALOResult(
                query=query, classification=classification, routing=routing,
                prompt=prompt, dry_run=True,
            )

        response = call_model(routing.model_id, prompt)
        return TALOResult(
            query=query, classification=classification, routing=routing,
            prompt=prompt, output=response["content"], dry_run=False,
            error=response["error"],
        )
