class Event:
    def execute(self, session):
        pass


class RunBatchEvent(Event):
    """Runs every active analysis once, in full detail, from start_cell -
    the batch/one-shot mode NBFix's CLI and MCP server use, as opposed
    to NBFix's incremental per-edit events (see its own events.py)."""

    def __init__(self, start_cell):
        self.start_cell = start_cell

    def execute(self, session):
        return session.run_analyses(self.start_cell, detailed=True)
