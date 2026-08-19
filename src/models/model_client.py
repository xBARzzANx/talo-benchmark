"""Unified model client using LiteLLM for all providers."""
import os, time, logging
from typing import Optional
import litellm
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MODEL_MAP = {
    "gpt-4.1-mini":      "gpt-4.1-mini",
    "claude-haiku-4-5":  "anthropic/claude-haiku-4-5-20251001",
    "gemini-2.5-flash":  "gemini/gemini-2.5-flash",
    "llama-3.1-8b":      "ollama/llama3.1:8b",
    "mistral-7b":        "ollama/mistral:7b-instruct",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-5",
}


def call_model(model_id: str, prompt: str, system_prompt: Optional[str] = None,
               temperature: float = 0.0, max_tokens: int = 2048, max_retries: int = 3) -> dict:
    """Call any benchmark model. Returns normalized response dict."""
    litellm_model = MODEL_MAP.get(model_id)
    if not litellm_model:
        raise ValueError(f"Unknown model_id: {model_id}")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(max_retries):
        try:
            start = time.time()
            response = litellm.completion(model=litellm_model, messages=messages,
                                          temperature=temperature, max_tokens=max_tokens)
            latency = time.time() - start
            return {
                "model_id": model_id,
                "content": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "latency_s": round(latency, 3),
                "error": None,
            }
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed for {model_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {"model_id": model_id, "content": None, "input_tokens": 0,
                        "output_tokens": 0, "latency_s": 0.0, "error": str(e)}
