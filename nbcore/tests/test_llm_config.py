import os
import unittest
from unittest.mock import patch

import pytest

pytest.importorskip("openai")

from nbcore.llm.config import DEFAULT_BASE_URL, DEFAULT_MODEL, default_client


class TestDefaultClient(unittest.TestCase):
    @patch("nbcore.llm.config.LLMClient")
    def test_defaults_to_local_ollama_without_api_key(self, mock_client_cls):
        with patch.dict(os.environ, {}, clear=True):
            default_client()
        mock_client_cls.assert_called_once_with(base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL)

    @patch("nbcore.llm.config.LLMClient")
    def test_passes_api_key_for_hosted_endpoint(self, mock_client_cls):
        env = {
            "NBCORE_LLM_BASE_URL": "https://llm.example.com/v1",
            "NBCORE_LLM_MODEL": "hosted-model",
            "NBCORE_LLM_API_KEY": "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            default_client()
        mock_client_cls.assert_called_once_with(
            base_url="https://llm.example.com/v1", model="hosted-model", api_key="secret",
        )


if __name__ == "__main__":
    unittest.main()
