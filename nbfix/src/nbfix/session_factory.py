from nbcore.session import AnalysisSession
from nbcore.analyses.dataleak_analysis import DataLeakAnalysis, DATA_LEAK

from .analyses.stale_cell_analysis import StaleCellAnalysis, STALE
from .analyses.idle_cell_analysis import IdleCellAnalysis, IDLE
from .analyses.isolated_cell_analysis import IsolatedCellAnalysis, ISOLATED


def new_session(level=5, filename="") -> AnalysisSession:
    """Builds an AnalysisSession with NBFix's own analysis set
    registered - the notebook-edit-lifecycle ones (stale/idle/isolated)
    plus data leakage. AnalysisSession itself knows nothing about which
    analyses exist; this is where NBFix opts into its four."""
    session = AnalysisSession(level=level, filename=filename)
    session.register_analysis(DATA_LEAK, DataLeakAnalysis())
    session.register_analysis(STALE, StaleCellAnalysis())
    session.register_analysis(IDLE, IdleCellAnalysis())
    session.register_analysis(ISOLATED, IsolatedCellAnalysis())
    return session
