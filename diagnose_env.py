import inspect
import os

import nbcore
import nbfix

print("nbcore package location:", nbcore.__file__)
print("nbfix package location:", nbfix.__file__)

from nbcore.llm.client import LLMClient
print("client.py location:", inspect.getsourcefile(LLMClient))

from nbfix.llm.api_sequence_prompts import API_SEQUENCE_SYSTEM_PROMPT
print("has attribution fix:", "cell_id must be the cell whose OWN code contains" in API_SEQUENCE_SYSTEM_PROMPT)

print("NBCORE_LLM_MODEL =", os.environ.get("NBCORE_LLM_MODEL"))
print("NBCORE_LLM_BASE_URL =", os.environ.get("NBCORE_LLM_BASE_URL"))
