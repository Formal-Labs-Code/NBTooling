import json

from tornado.httpclient import HTTPClientError

TEST_NOTEBOOK = [
    {"cell_type": "code", "source": "x = 1"},
    {"cell_type": "code", "source": "y = x + 1"},
]


async def _post_event(jp_fetch, event, notebook_id, params=None):
    try:
        response = await jp_fetch(
            "nbfix",
            "api",
            "event",
            method="POST",
            body=json.dumps(
                {"event": event, "notebook_id": notebook_id, "params": params or {}}
            ),
        )
    except HTTPClientError as exc:
        return exc.response.code, json.loads(exc.response.body)
    return response.code, json.loads(response.body)


async def test_open_notebook_returns_diagnostics(jp_fetch):
    code, payload = await _post_event(
        jp_fetch, "open_notebook", "nb1.ipynb", {"notebook_json": TEST_NOTEBOOK}
    )
    assert code == 200
    assert payload["status"] == "success"
    assert "diagnostics" in payload


async def test_unknown_event_returns_400(jp_fetch):
    notebook_id = "nb1.ipynb"
    await _post_event(
        jp_fetch, "open_notebook", notebook_id, {"notebook_json": TEST_NOTEBOOK}
    )
    code, payload = await _post_event(jp_fetch, "not_a_real_event", notebook_id)
    assert code == 400
    assert payload["status"] == "error"


async def test_missing_required_field_returns_400(jp_fetch):
    try:
        response = await jp_fetch(
            "nbfix",
            "api",
            "event",
            method="POST",
            body=json.dumps({"event": "open_notebook"}),
        )
        code, body = response.code, response.body
    except HTTPClientError as exc:
        code, body = exc.response.code, exc.response.body
    assert code == 400
    payload = json.loads(body)
    assert payload["status"] == "error"


async def test_event_without_open_session_returns_404(jp_fetch):
    code, payload = await _post_event(
        jp_fetch, "run_cell", "never-opened.ipynb", {"cell_index": 0}
    )
    assert code == 404
    assert payload["status"] == "error"


async def test_run_cell_after_open(jp_fetch):
    notebook_id = "nb2.ipynb"
    await _post_event(
        jp_fetch, "open_notebook", notebook_id, {"notebook_json": TEST_NOTEBOOK}
    )
    code, payload = await _post_event(
        jp_fetch, "run_cell", notebook_id, {"cell_index": 0}
    )
    assert code == 200
    assert payload["status"] == "success"


async def test_close_notebook_evicts_session(jp_fetch):
    notebook_id = "nb3.ipynb"
    await _post_event(
        jp_fetch, "open_notebook", notebook_id, {"notebook_json": TEST_NOTEBOOK}
    )
    close_code, _ = await _post_event(jp_fetch, "close_notebook", notebook_id)
    assert close_code == 200

    code, payload = await _post_event(
        jp_fetch, "run_cell", notebook_id, {"cell_index": 0}
    )
    assert code == 404
