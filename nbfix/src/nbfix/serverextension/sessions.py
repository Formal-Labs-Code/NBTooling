from nbcore.session import AnalysisSession

from ..session_factory import new_session


class SessionStore:
    """
    Holds one AnalysisSession per open notebook, keyed by notebook id
    (the notebook's server-side path).
    """

    def __init__(self):
        self._sessions: dict[str, AnalysisSession] = {}

    def open(self, notebook_id: str) -> AnalysisSession:
        """A fresh session for the notebook, replacing any earlier one - a
        page reload or crashed tab never sends close_notebook, and an
        AnalysisSession can only be loaded once."""
        self._sessions[notebook_id] = new_session()
        return self._sessions[notebook_id]

    def get(self, notebook_id: str):
        return self._sessions.get(notebook_id)

    def close(self, notebook_id: str) -> None:
        self._sessions.pop(notebook_id, None)
