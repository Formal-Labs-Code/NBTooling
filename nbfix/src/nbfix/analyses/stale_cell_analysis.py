from collections import defaultdict
from copy import deepcopy
from nbcore.ir.intermediate_representations import IntermediateRepresentations
from nbcore.analyses.runner.runners import Runner
from nbcore.analyses.runner.analyses_utils import find_changed_vars, AssignParserVisitor, get_all_unbound_vars
from nbcore.analyses.analysis import Analysis
from .abs_states.code_impact_abs_state import CodeImpactAS
from nbcore.analyses.runner.analysis_results import Result
from nbcore.analyses.runner.stats import Stats
from nbcore.parser.cfg_nodes import Node
from nbcore.parser import ast_nodes as ast

STALE = "Stale Cells Analysis"


class StaleCellAnalysis(Analysis):
    def __init__(self) -> None:
        super().__init__()
        self.k = 2
        self.abstract_state = CodeImpactAS()
        self.stats = []

    def F_transformer(self, cfg_node: Node, a_state: CodeImpactAS, cell_IR: IntermediateRepresentations):
        as_transformed: CodeImpactAS = deepcopy(a_state)
        if not cfg_node.ast_node:
            return a_state
        if isinstance(cfg_node.ast_node, ast.Assign):
            assign_parser = AssignParserVisitor()
            assign_parser.parse_assign(cfg_node.ast_node)
            def_vars = assign_parser.def_variables
            assigned_vars = assign_parser.assigned_variables
            if len(def_vars):
                for def_var in def_vars:
                    for var in assigned_vars & as_transformed.impacted_variables.keys() - def_vars - self.imports:
                        if as_transformed.impacted_variables[var] != -1:
                            if var in cell_IR.UDA.def_use_chains.unbound_names and var not in cell_IR.UDA.defined_vars:
                                as_transformed.set_var_level(def_var, as_transformed.impacted_variables[var] + 1)
                            else:
                                as_transformed.set_var_level(def_var, as_transformed.impacted_variables[var])

        if isinstance(cfg_node.ast_node, ast.Name):
            if cfg_node.label in as_transformed.impacted_variables.keys() and as_transformed.impacted_variables[cfg_node.label] != -1:
                if cfg_node.label in cell_IR.UDA.def_use_chains.unbound_names:
                    as_transformed.set_var_level(cfg_node.label + "_usage", as_transformed.impacted_variables[cfg_node.label] + 1)
                else:
                    as_transformed.set_var_level(cfg_node.label + "_usage", as_transformed.impacted_variables[cfg_node.label])
        return as_transformed

    def combine_states(self, states: list[CodeImpactAS]):
        res_state = CodeImpactAS(K = self.k)
        for s in states:
            res_state.aug_join(s)
        return res_state

    def summarize_result(self, result: Result) -> Result:
        summarized_result: Result = result.distinct_errors()
        return summarized_result

    def _prepare_init_as(self, new_cell_IR, old_cell_IR):
        init_as: CodeImpactAS = deepcopy(self.abstract_state)
        changed_vars: set[str] = find_changed_vars(new_cell_IR, old_cell_IR)
        for var in changed_vars:
            init_as.impacted_variables[var] = 0
        for var in new_cell_IR.UDA.defined_vars.keys():
            if (var in init_as.impacted_variables and init_as.impacted_variables[var] > 0):
                init_as.impacted_variables[var] = 0
        return init_as

    # Hard ceiling on inter-cell fixpoint recursion depth, enforced here
    # rather than trusted from callers: inter_fixpoint_runner's branching
    # factor is the number of cells sharing a variable with the notebook
    # (can be dozens on a real notebook), so total recursive calls scale
    # roughly as branching_factor**level - at level=5 this was observed to
    # hit 184k+ recursive calls within 20s (still climbing) on a real
    # notebook (see scripts/repro_stale_hang.py). update_abstract_state
    # below was found calling _run_fixpoint_analysis without passing
    # `level` at all, silently defaulting to the old level=20 - clamping
    # here fixes both call paths at once instead of relying on every
    # caller threading the right value through.
    #
    # level=2 was tried first but is too shallow to detect staleness
    # through even a 2-hop dependency chain (x -> y -> z: editing x and
    # re-running produced zero findings at level=2, correctly flagged z
    # at level=3+). level=3 looked like the minimum needed based on that
    # 2-hop check alone, but running the full test suite against it broke
    # a pre-existing test (test_stale_cell_analysis.py::test_analyze_notebook,
    # Test.ipynb) that depends on a 3-hop chain reaching cell 5 - level=3
    # wasn't actually sufficient, it just happened to be sufficient for the
    # one manual case checked at the time. level=4 is the minimum that
    # passes the full existing suite, and still empirically fast on the
    # dense real notebook this was fixing in the first place (9750 calls,
    # 0.11s at level=4 vs 184k+/20s+ at level=5). The real fix is
    # tightening CodeImpactAS.contains()/projection() pruning so depth
    # doesn't have to trade off against branching-factor safety at all;
    # this is a stopgap, not that fix.
    MAX_LEVEL = 4

    def _run_fixpoint_analysis(self, notebook_IR, old_cell_IR=None, level=20, filename=""):
        level = min(level, self.MAX_LEVEL)
        self._find_all_imports(notebook_IR)
        self.all_unbound_vars = get_all_unbound_vars(notebook_IR)
        init_as: CodeImpactAS = self._prepare_init_as(
            notebook_IR[old_cell_IR.cell_id], old_cell_IR
        )
        stat = Stats(old_cell_IR.cell_id, filename)
        stat.log_start()
        runner: Runner = Runner(stat, defaultdict(CodeImpactAS), notebook_IR)
        result: Result = runner.inter_fixpoint_runner(
            self,
            old_cell_IR.cell_id,
            abstract_state=init_as,
            K=level,
            cpath=[],
            results=Result()
        )

        stat.log_end()
        self.stats.append(stat)
        return runner, result, init_as

    def _find_all_imports(self, notebook_IR: dict[str, IntermediateRepresentations]):
        self.imports = set()
        for cell_IR in notebook_IR.values():
            self.imports.update(cell_IR.UDA.imports)

    def analyze_notebook(self, notebook_IR, old_cell_IR=None, level=20, filename=""):
        return self._run_fixpoint_analysis(notebook_IR, old_cell_IR, level, filename)[1]

    def update_abstract_state(self, cell_IR, notebook_IR):
        self.find_necessary_cells(notebook_IR)
        old_cell_IR: IntermediateRepresentations = IntermediateRepresentations(
            cell_IR.last_ran_code, cell_IR.cell_id
        )
        runner, _, init_as = self._run_fixpoint_analysis(notebook_IR, old_cell_IR)
        for abs in runner.error_states:
            for var, level in abs.impacted_variables.items():
                if level == 1:
                    self.abstract_state.impacted_variables[var] = level
        for var, level in init_as.impacted_variables.items():
            if level == 0:
                self.abstract_state.impacted_variables[var] = -1

    def phi_condition(self, current: set, pre: set, cell_IR):
        if len(self.all_unbound_vars.intersection(cell_IR.UDA.defined_vars.keys())):
            return pre <= current
        else:
            False

    def calculate_pre(self, cell_IR:IntermediateRepresentations):
        return cell_IR.UDA.unbound_final - self.imports