from collections import defaultdict
from copy import deepcopy

from nbcore.events import Event
from nbcore.ir.intermediate_representations import IntermediateRepresentations
from nbcore.analyses.dataleak_analysis import DATA_LEAK
from nbcore.resource_utils.utils import load_notebook

from .analyses.stale_cell_analysis import STALE
from .analyses.idle_cell_analysis import IDLE
from .analyses.isolated_cell_analysis import ISOLATED


def reindex_results(session, updated_index: int, added=True):
    for result in session.results.values():
        for path_result in result.path_results:
            for error in path_result.error_infos:
                if added and error.cell_id >= updated_index:
                    error.cell_id += 1
                    path_result.path = [error.cell_id]
                if not added and error.cell_id > updated_index:
                    error.cell_id -= 1
                    path_result.path = [error.cell_id]


class AddActiveAnalysesEvent(Event):
    def __init__(self, active_analyses):
        self.active_analyses = active_analyses

    def execute(self, session):
        session.add_analyses(self.active_analyses)
        return session.run_analyses(-1, [IDLE, ISOLATED]).join_by_cell_id()


class OpenNotebookEvent(Event):
    def __init__(self, notebook_json):
        self.notebook_json = notebook_json

    def execute(self, session):
        session.load(load_notebook(self.notebook_json))


class RunCellEvent(Event):
    def __init__(self, cell_index):
        self.cell_index = cell_index

    def execute(self, session):
        results = session.run_analyses(self.cell_index, [STALE, DATA_LEAK])
        session.update_abstract_states(self.cell_index)
        session.cells[self.cell_index].last_ran_code = session.cells[self.cell_index].cell_code
        return results.join_by_cell_id()


class AddCellEvent(Event):
    def __init__(self, position: int, kind: int, content: str) -> None:
        self.position: int = int(position)
        self.kind = int(kind)
        self.content = str(content)

    def execute(self, session):
        new_cells: dict[int, IntermediateRepresentations] = defaultdict()
        all_keys = list(session.cells.keys())
        max_key = max(all_keys)
        for i in range(len(all_keys)):
            if all_keys[i] >= self.position:
                new_cells[all_keys[i] + 1] = session.cells[all_keys[i]]
                new_cells[all_keys[i] + 1].cell_id = all_keys[i] + 1
            if all_keys[i] < self.position:
                new_cells[all_keys[i]] = session.cells[all_keys[i]]

        if self.position <= max_key:
            new_cells[max_key + 1] = session.cells[max_key]
            new_cells[max_key + 1].cell_id = max_key + 1

        if self.kind == 2:
            new_cells[self.position] = IntermediateRepresentations(self.content, self.position)
        session.cells = new_cells
        reindex_results(session, self.position)
        return session.run_analyses(-1, [IDLE, ISOLATED]).join_by_cell_id()


class RemoveCellEvent(Event):
    def __init__(self, position: int) -> None:
        self.position: int = position

    def execute(self, session):
        new_cells: dict[int, IntermediateRepresentations] = defaultdict()
        for index in session.cells.keys():
            if index > self.position:
                new_cells[index - 1] = session.cells[index]
                new_cells[index - 1].cell_id = index - 1
            if index < self.position:
                new_cells[index] = session.cells[index]

        session.cells = deepcopy(new_cells)
        reindex_results(session, self.position, False)
        return session.run_analyses(-1, [IDLE, ISOLATED]).join_by_cell_id()


class ChangeCellCodeEvent(Event):
    def __init__(self, new_code: str, cell_index: int, with_result: bool) -> None:
        self.new_code: str = new_code
        self.cell_index: int = cell_index
        self.with_result: bool = with_result

    def execute(self, session):
        if self.cell_index in session.cells.keys():
            last_ran_code = session.cells[self.cell_index].last_ran_code
            session.cells[self.cell_index] = IntermediateRepresentations(self.new_code, self.cell_index, last_ran_code)
        if self.with_result:
            return session.run_analyses(-1, [IDLE, ISOLATED]).join_by_cell_id()


class CloseNotebookEvent(Event):
    def execute(self, session):
        session.reset()
