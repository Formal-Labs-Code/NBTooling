import os

from .client import LLMClient

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5-coder:14b"


def default_client() -> LLMClient:
    """
    Builds an LLMClient from NBCORE_LLM_BASE_URL/NBCORE_LLM_MODEL env
    vars, defaulting to a local Ollama instance. One shared prefix, not
    one per tool: this function is nbcore-owned and used identically by
    every LLM feature (see detect_bugs_event.py,
    detect_stale_cells_event.py, detect_api_sequence_event.py) - a
    per-tool prefix would need this function parameterized per caller,
    which nothing currently needs.
    """
    return LLMClient(
        base_url=os.environ.get("NBCORE_LLM_BASE_URL", DEFAULT_BASE_URL),
        model=os.environ.get("NBCORE_LLM_MODEL", DEFAULT_MODEL),
    )
