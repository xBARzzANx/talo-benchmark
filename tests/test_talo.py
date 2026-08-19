"""Unit tests for the TALO package: classifier, router, prompt selector,
and orchestrator."""
import pytest

from talo.classifier import TaskClassifier
from talo.router import TALORouter, SUPPORTED_TASK_CLASSES, FALLBACK_TASK_CLASS
from talo.prompt_selector import PromptSelector, PromptSelectionError
from talo.orchestrator import TALOOrchestrator


# --- Classifier ------------------------------------------------------------

@pytest.fixture
def classifier():
    return TaskClassifier()


@pytest.mark.parametrize("query,expected", [
    ("Show me total revenue by region from the orders table.", "sql_generation"),
    ("Which customers have unusually high return rates?", "sql_generation"),
    ("Here is last week's daily order volume: 1240, 1198, 1305, 1276, 4890, "
     "1241, 1189. Anything I should look at?", "anomaly_detection"),
    ("Our conversion rate is 2.1% this quarter, down from 2.6%. "
     "What does that mean for us?", "kpi_interpretation"),
    ("Summarise the key figures for the board.", "report_generation"),
])
def test_classify_clear_cases(classifier, query, expected):
    result = classifier.classify(query)
    assert result.task_class == expected
    assert not result.used_fallback


def test_classify_falls_back_on_low_evidence(classifier):
    result = classifier.classify("Churn went up again.")
    assert result.used_fallback
    assert result.task_class == FALLBACK_TASK_CLASS


def test_classify_empty_query_falls_back(classifier):
    result = classifier.classify("")
    assert result.used_fallback
    assert result.task_class == FALLBACK_TASK_CLASS


def test_classify_confidence_in_unit_interval(classifier):
    result = classifier.classify("Show me revenue by region.")
    assert 0.0 <= result.confidence <= 1.0


# --- Router ------------------------------------------------------------

@pytest.fixture
def router():
    return TALORouter()


def test_router_supports_four_task_classes(router):
    assert set(SUPPORTED_TASK_CLASSES) == {
        "sql_generation", "anomaly_detection", "kpi_interpretation", "report_generation",
    }


def test_router_matches_alignment_matrix(router):
    expected = {
        "sql_generation": ("gemini-2.5-flash", "few_shot"),
        "anomaly_detection": ("gpt-4.1-mini", "few_shot"),
        "kpi_interpretation": ("claude-haiku-4-5", "chain_of_thought"),
        "report_generation": ("gpt-4.1-mini", "few_shot"),
    }
    for task_class, (model_id, strategy) in expected.items():
        decision = router.route(task_class)
        assert decision.model_id == model_id
        assert decision.strategy == strategy


def test_router_rejects_unknown_task_class(router):
    with pytest.raises(ValueError):
        router.route("not_a_real_task_class")


def test_router_route_all_covers_every_class(router):
    decisions = router.route_all()
    assert set(decisions.keys()) == set(SUPPORTED_TASK_CLASSES)


# --- Prompt selector ------------------------------------------------------------

@pytest.fixture
def prompt_selector():
    return PromptSelector()


def test_prompt_selector_zero_shot_fills_slots(prompt_selector):
    prompt = prompt_selector.build_prompt(
        "sql_generation", "zero_shot",
        schema="orders(id INT)", question="How many orders?",
    )
    assert "orders(id INT)" in prompt
    assert "How many orders?" in prompt


def test_prompt_selector_injects_few_shot_examples(prompt_selector):
    prompt = prompt_selector.build_prompt(
        "sql_generation", "few_shot",
        schema="orders(id INT)", question="How many orders?",
    )
    assert "Example 1" in prompt


def test_prompt_selector_explicit_examples_override_default(prompt_selector):
    prompt = prompt_selector.build_prompt(
        "anomaly_detection", "few_shot", data="0,1\n1,2", examples="CUSTOM EXAMPLE",
    )
    assert "CUSTOM EXAMPLE" in prompt
    assert "Example 1" not in prompt


def test_prompt_selector_unknown_task_class_raises(prompt_selector):
    with pytest.raises(PromptSelectionError):
        prompt_selector.build_prompt("not_a_real_task_class", "zero_shot")


def test_prompt_selector_missing_slot_raises(prompt_selector):
    with pytest.raises(PromptSelectionError):
        prompt_selector.build_prompt("sql_generation", "zero_shot", schema="orders(id INT)")


# --- Orchestrator ------------------------------------------------------------

@pytest.fixture
def orchestrator():
    return TALOOrchestrator()


def test_orchestrator_dry_run_makes_no_api_call(orchestrator, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("call_model must not be invoked in a dry run")

    monkeypatch.setattr("talo.orchestrator.call_model", fail_if_called)

    result = orchestrator.run("Show me revenue by region.")
    assert result.dry_run is True
    assert result.output is None
    assert result.prompt is not None
    assert result.error is None


def test_orchestrator_dry_run_routes_to_expected_model(orchestrator):
    result = orchestrator.run("Show me revenue by region.")
    assert result.classification.task_class == "sql_generation"
    assert result.routing.model_id == "gemini-2.5-flash"


def test_orchestrator_live_run_invokes_call_model(orchestrator, monkeypatch):
    calls = []

    def fake_call_model(model_id, prompt, *args, **kwargs):
        calls.append((model_id, prompt))
        return {"content": "mock output", "error": None}

    monkeypatch.setattr("talo.orchestrator.call_model", fake_call_model)

    result = orchestrator.run("Show me revenue by region.", dry_run=False)
    assert len(calls) == 1
    assert result.output == "mock output"
    assert result.dry_run is False
