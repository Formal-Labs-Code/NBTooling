import functools
import json
import os

import tornado
from tornado.ioloop import IOLoop
from jupyter_server.base.handlers import APIHandler
from jupyter_server.utils import url_path_join

from nbcore.analyses.dataleak_analysis import DATA_LEAK

from ..analyses.idle_cell_analysis import IDLE
from ..analyses.isolated_cell_analysis import ISOLATED
from ..analyses.stale_cell_analysis import STALE
from .dispatch import InvalidEventError, build_event
from .sessions import SessionStore

DEFAULT_ANALYSES = [DATA_LEAK, STALE, IDLE, ISOLATED]


class _NBFixHandler(APIHandler):
    def _respond_error(self, status_code, message):
        self.set_status(status_code)
        self.finish(json.dumps({"status": "error", "message": message}))


class EventHandler(_NBFixHandler):
    """
    Single endpoint that receives {event, notebook_id, params} and dispatches
    it to the matching Event on the AnalysisSession for that notebook.
    """

    @tornado.web.authenticated
    def post(self):
        body = self.get_json_body() or {}
        try:
            event_name = body["event"]
            notebook_id = body["notebook_id"]
        except KeyError as exc:
            self._respond_error(400, f"Missing required field: {exc}")
            return

        params = body.get("params") or {}
        sessions: SessionStore = self.settings["nbfix_sessions"]

        if event_name == "open_notebook":
            session = sessions.open(notebook_id)
        else:
            session = sessions.get(notebook_id)
            if session is None:
                self._respond_error(404, f"No open session for notebook {notebook_id!r}")
                return

        try:
            event = build_event(event_name, params)
        except InvalidEventError as exc:
            self._respond_error(400, str(exc))
            return

        try:
            result = session.execute_event(event)

            if event_name == "open_notebook":
                # Bootstrap the default set of active analyses so the first
                # response already carries diagnostics, mirroring what the
                # VS Code prototype did via a follow-up add_active_analyses call.
                analyses_event = build_event("add_active_analyses", {"active_analyses": DEFAULT_ANALYSES})
                result = session.execute_event(analyses_event)
        except Exception as exc:
            # Catches failures that only surface at execution time rather
            # than at dispatch (e.g. an unreachable local LLM endpoint for
            # detect_bugs) - one event failing this way must not crash the
            # request with an unhandled-exception response.
            self._respond_error(500, f"Event execution failed: {exc}")
            return

        if event_name == "close_notebook":
            sessions.close(notebook_id)

        # checked_cells is specific to DetectApiSequenceEvent (a plain
        # attribute set during execute(), not part of Result - see that
        # class's docstring for why) - getattr keeps every other event
        # untouched rather than needing every Event subclass to carry a
        # checked_cells attribute just so this line can read it uniformly.
        checked_cells = getattr(event, "checked_cells", None)
        self._respond_success(result, checked_cells)

    def _respond_success(self, result, checked_cells=None):
        diagnostics = []
        # Result.dumps() returns '' (not '[]') when there are no findings -
        # json.loads('') would raise, so only attempt to parse when there's
        # actually something to parse. A clean "no findings" result is the
        # common case for detect_bugs (most checks won't find a bug), not
        # an edge case to special-case away.
        if result is not None and result.path_results:
            try:
                diagnostics = json.loads(result.dumps())
            except (ValueError, AttributeError):
                self._respond_error(500, "Failed to serialize analysis results")
                return
        response = {"status": "success", "diagnostics": diagnostics}
        if checked_cells is not None:
            response["checked_cells"] = sorted(checked_cells)
        self.finish(json.dumps(response))


class FixHandler(_NBFixHandler):
    """
    Suggests a repair for every data leak in the notebook sent with the
    request - {"cells": [{"cell_type", "source"}], "path"?} - and responds
    {"status": "success", "fix": <nbfix.repair.fix result>}.

    Not an EventHandler event: it neither reads nor changes the notebook's
    session (the frontend sends the cells as they are now, so the repair is
    never based on stale server-side state), and an LLM repair takes tens of
    seconds, so it runs in a worker thread rather than blocking the server.
    """

    @tornado.web.authenticated
    async def post(self):
        body = self.get_json_body() or {}
        cells = body.get("cells")
        if not isinstance(cells, list) or not all(
            isinstance(cell, dict) and isinstance(cell.get("cell_type"), str) and isinstance(cell.get("source"), str)
            for cell in cells
        ):
            self._respond_error(400, 'cells must be a list of {"cell_type", "source"}')
            return

        from ..repair import fix

        notebook = {"cells": [
            {"cell_type": cell["cell_type"], "source": cell["source"], "metadata": {}} for cell in cells
        ]}
        name = os.path.basename(body.get("path") or "") or None
        try:
            result = await IOLoop.current().run_in_executor(
                None, functools.partial(fix, notebook, filename=name)
            )
        except Exception as exc:
            # An unreachable/missing LLM, or the analysis failing on this
            # notebook - either way, a clean error rather than a crash.
            self._respond_error(500, f"Fix failed: {exc}")
            return
        self.finish(json.dumps({"status": "success", "fix": result}))


def setup_handlers(web_app):
    host_pattern = ".*$"
    base_url = web_app.settings["base_url"]
    web_app.settings.setdefault("nbfix_sessions", SessionStore())
    web_app.add_handlers(host_pattern, [
        (url_path_join(base_url, "nbfix", "api", "event"), EventHandler),
        (url_path_join(base_url, "nbfix", "api", "fix"), FixHandler),
    ])
