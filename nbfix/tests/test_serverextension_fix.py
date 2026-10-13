import json
from unittest.mock import MagicMock, patch

import pytest
from tornado.httpclient import HTTPClientError

pytest.importorskip("openai")

from nbcore.llm.client import LLMClientError
from nbcore.resource_utils.utils import TEST_RES_PATH, read_json

FIXED_CELL_2 = "lr = LogisticRegression()\na = lr.fit(X_train, y_train)"


def _cells(name):
    return [
        {"cell_type": cell["cell_type"], "source": "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]}
        for cell in read_json(TEST_RES_PATH + name)["cells"]
    ]


async def _post_fix(jp_fetch, body, *responses):
    client = MagicMock()
    client.chat_json.side_effect = list(responses)
    with patch("nbcore.llm.config.default_client", return_value=client):
        try:
            response = await jp_fetch("nbfix", "api", "fix", method="POST", body=json.dumps(body))
        except HTTPClientError as exc:
            return exc.response.code, json.loads(exc.response.body)
    return response.code, json.loads(response.body)


async def test_fix_returns_verified_repair(jp_fetch):
    code, payload = await _post_fix(
        jp_fetch, {"cells": _cells("dataleak_overlap_true.ipynb"), "path": "work/leaky.ipynb"},
        {"edits": [{"cell": 2, "code": FIXED_CELL_2}], "explanation": "train on the split"},
    )
    assert code == 200
    assert payload["status"] == "success"
    fix = payload["fix"]
    assert fix["status"] == "verified"
    assert fix["edits"] == [{"cell": 2, "code": FIXED_CELL_2}]
    assert "--- a/leaky.ipynb (cell 2)" in fix["diff"]


async def test_fix_clean_notebook_has_no_findings(jp_fetch):
    code, payload = await _post_fix(jp_fetch, {"cells": _cells("dataleak_overlap_false.ipynb")})
    assert code == 200
    assert payload["fix"]["status"] == "no_findings"


async def test_fix_rejects_malformed_cells(jp_fetch):
    code, payload = await _post_fix(jp_fetch, {"cells": [{"source": "x = 1"}]})
    assert code == 400
    assert payload["status"] == "error"


async def test_fix_unreachable_llm_returns_500(jp_fetch):
    code, payload = await _post_fix(
        jp_fetch, {"cells": _cells("dataleak_overlap_true.ipynb")},
        LLMClientError("Could not reach LLM endpoint"),
    )
    assert code == 500
    assert payload == {"status": "error", "message": "Fix failed: Could not reach LLM endpoint"}
