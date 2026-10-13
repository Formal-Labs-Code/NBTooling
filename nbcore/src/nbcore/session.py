from collections import defaultdict

from .analyses.runner.analysis_results import Result, PathResult, ErrorType, ErrorInfo
from .ir.intermediate_representations import IntermediateRepresentations


class AnalysisSession:
    """
    Generic driver: holds a dict of cells (whichever unit a tool loads -
    notebook cells, or a script's `# %%` cells), a pluggable registry of
    analyses, and the run/event-dispatch machinery every tool needs on
    top of that. No tool-specific loading or analysis set is hardcoded
    here - each tool supplies its own via register_analysis()/load().
    """

    def __init__(self, level=5, filename=""):
        self.reset()
        self.all_analyses: dict[str, object] = {}
        self.level = level
        self.results: dict[str, Result] = defaultdict(Result)
        self.filename = filename

    def register_analysis(self, name: str, analysis) -> None:
        self.all_analyses[name] = analysis

    def load(self, cells: dict[int, IntermediateRepresentations]) -> None:
        if self.cells:
            raise Exception("A notebook or script has already been loaded.")
        self.results: dict[str, Result] = defaultdict(Result)
        self.cells = cells

    def add_analyses(self, analyses):
        '''
        Add analyses to dict of active analyses.

        Parameters
        ----------
        analyses: list(str)
            A list of analyses to be added to active analyses.
        '''
        self.active_analyses = analyses

    def update_abstract_states(self, cell_index):
        for analysis in self.all_analyses.values():
            analysis.update_abstract_state(self.cells[cell_index], self.cells)

    def execute_event(self, event):
        return event.execute(self)

    def join_analyses_results(self):
        new_results: Result = Result()
        for analysis_str in self.active_analyses:
            new_results.join_results(self.results[analysis_str])
        return new_results

    def run_analyses(self, cell_index: int, analyses: list[str] = None, detailed: bool = False):
        if cell_index == -1:
            changed_cell_IR = None
        else:
            if (cell_index in self.cells):
                changed_cell_IR = IntermediateRepresentations(self.cells[cell_index].last_ran_code, cell_index)
            else:
                new_results: Result = Result()
                new_results.add_path_results([PathResult([cell_index], [ErrorInfo(cell_index, 0, "", ErrorType.CRITICAL, "Cannot start from cell with no code")])])
                return new_results
        if not analyses:
            analyses = self.active_analyses
        for analysis_str in analyses:
            if analysis_str in self.active_analyses:
                self.all_analyses[analysis_str].find_necessary_cells(self.cells)
                self.results[analysis_str] = self.all_analyses[analysis_str].analyze_notebook(self.cells, changed_cell_IR, self.level, self.filename)
                if not detailed:
                    self.results[analysis_str] = self.all_analyses[analysis_str].summarize_result(self.results[analysis_str])
        return self.join_analyses_results()

    def reset(self):
        self.cells: dict[int, IntermediateRepresentations] | None = None
        self.active_analyses: list[str] = []

    def __str__(self) -> str:
        cells = f""
        for i, cell in self.cells.items():
            cells += f"[{i}:{cell}, last  run: {self.cells[i].last_ran_code}]"
        return f"Cells: {cells}\n Analyses: {self.active_analyses}"
