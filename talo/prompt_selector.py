"""
Implements the prompt strategy selection and slot-filling component
described in Section 5.4.

Looks up the prompt template for a given task class and strategy, injects
the few-shot examples from src/prompts/examples.py when strategy is
"few_shot", and fills the template slots from the task input.
"""
from src.prompts.templates import TEMPLATES, get_prompt
from src.prompts.examples import FEW_SHOT_EXAMPLES


class PromptSelectionError(Exception):
    """Raised when a template or a required slot is missing."""


class PromptSelector:
    """
    Resolves a (task_class, strategy) pair to a fully rendered prompt,
    filling template slots from task input and auto-injecting few-shot
    examples where the strategy requires them.
    """

    def build_prompt(self, task_class: str, strategy: str, **slot_values) -> str:
        """
        Render the prompt for a task class and prompt strategy.

        Args:
            task_class: One of the four supported task classes.
            strategy: One of "zero_shot", "few_shot", "chain_of_thought",
                "structured_output".
            **slot_values: Template slot values (e.g. schema, question, data).
                For "few_shot", an "examples" slot is auto-injected from
                src/prompts/examples.py if not explicitly provided.

        Returns:
            The rendered prompt string.

        Raises:
            PromptSelectionError: If the (task_class, strategy) combination
                has no template, or a required slot is missing.
        """
        if task_class not in TEMPLATES:
            raise PromptSelectionError(f"Unknown task class: '{task_class}'")
        if strategy not in TEMPLATES[task_class]:
            raise PromptSelectionError(
                f"No '{strategy}' template for task class '{task_class}'"
            )

        if strategy == "few_shot" and "examples" not in slot_values:
            slot_values["examples"] = FEW_SHOT_EXAMPLES.get(task_class, "")

        try:
            return get_prompt(task_class, strategy, **slot_values)
        except KeyError as e:
            raise PromptSelectionError(
                f"Missing slot {e} for '{task_class}' / '{strategy}' template"
            ) from e
