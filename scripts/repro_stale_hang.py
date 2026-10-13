"""
Reproduces the StaleCellAnalysis non-termination found while manually testing
the JupyterLab extension: RunCellEvent on victim.ipynb's cell 2 (which
defines `df`, read by most of the rest of the notebook) causes
Runner.inter_fixpoint_runner to recurse explosively - 184k+ calls in 20s
and still climbing, no sign of terminating. Reproduces purely in-process,
no jupyter_server/HTTP involved, so it's a core algorithm issue in the
inter-cell fixpoint propagation (phi_condition / CodeImpactAS.contains()
lattice comparison in runners.py + abs_states/code_impact_abs_state.py),
not a server-concurrency bug.

Run with: python scripts/repro_stale_hang.py [timeout_seconds]
"""
import json
import signal
import sys
import time

from nbcore.analyses.runner.runners import Runner
from nbcore.resource_utils.utils import load_notebook
from nbfix.events import RunCellEvent
from nbfix.session_factory import new_session


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def main():
    timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    call_count = [0]
    orig = Runner.inter_fixpoint_runner

    def counted(self, *a, **kw):
        call_count[0] += 1
        return orig(self, *a, **kw)

    Runner.inter_fixpoint_runner = counted

    nb = json.load(open("tests/resources/victim.ipynb"))
    cells = [{"cell_type": c["cell_type"], "source": "".join(c["source"])} for c in nb["cells"]]

    nbfix = new_session(level=5)
    nbfix.load(load_notebook(cells))
    nbfix.add_analyses(["Stale Cells Analysis"])

    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    t0 = time.time()
    try:
        RunCellEvent(2).execute(nbfix)
        print(f"DONE in {time.time() - t0:.2f}s, inter_fixpoint_runner calls={call_count[0]}")
    except Timeout:
        print(f"STILL RUNNING after {time.time() - t0:.2f}s, "
              f"inter_fixpoint_runner calls={call_count[0]} so far")
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
