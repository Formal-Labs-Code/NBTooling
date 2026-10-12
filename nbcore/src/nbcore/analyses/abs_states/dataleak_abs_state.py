from __future__ import annotations
from ...parser import ast_nodes as ast
from ...parser.cfg_nodes import Node, AssignmentNode
from ...ir.intermediate_representations import IntermediateRepresentations
from .abs_state import AbstractState
from ..abs_domains.dataleak_lattice.data_frame_sets import DataFrameSet
from ..runner.analysis_results import ErrorInfo, ErrorType
from itertools import product
import re
from enum import Enum, auto

_MANGLED = re.compile(r"~inline_\d+_\w+")


def demangle(text, cell_IR) -> tuple:
    '''
    (text with inlined-call names restored to their originals, name of the
    inlined function they came from or None), e.g.
    "~inline_0_test_model_model.predict" -> ("model.predict", "test_model").
    '''
    if text is None:
        return None, None
    names = getattr(getattr(cell_IR, "CFG", None), "inline_names", None) or {}
    found = []

    def restore(match):
        original = names.get(match.group(0))
        if original is None:
            return match.group(0)
        found.append(original[1])
        return original[0]

    return _MANGLED.sub(restore, text), (found[0] if found else None)


def _where(site: dict) -> str:
    '''"(in `f`, cell 2, line 5)" / "(cell 2, line 5)" for a recorded site.'''
    scope = f"in `{site['function']}`, " if site.get("function") else ""
    return f"({scope}cell {site['cell']}, line {site['line']})"


class Usage(Enum):
    TRAIN = auto()
    TEST = auto()
    NONE = auto()

class DataLeakAbstractDomain:
    def __init__(self, dfs: DataFrameSet = None, taint: bool = False, usages: set[Usage] = None, dull=False, fitted_from: tuple = None, taint_origin: dict = None, usage_sites: dict = None) -> None:
        '''
        Abstract domain that keeps track of data frames, if those data frames are tainted,
        if any data frames have been used for testing or training
        '''
        self.dull = dull
        self.dfs: DataFrameSet = dfs if dfs is not None else DataFrameSet()
        self.taint: bool = taint
        self.usages: set[Usage] = usages if usages is not None else set[Usage]()
        # Set on a RECEIVER variable (e.g. "scaler" in `scaler.fit(X)`),
        # not on an assigned result - (source_name, was_full_rows_at_fit_time,
        # fit_origin), fit_origin being the taint_origin-shaped location of
        # the fit() call itself.
        # Consulted by DataLeakAnalysis.transform_transformer to tell "fit on
        # the whole, unsplit dataset, transform() separately per split" (a
        # real leak) apart from "fit on an already-split subset, transform()
        # both splits" (the correct, recommended pattern) - both look
        # identical as a call sequence, so this is what actually
        # distinguishes them. See dataleak_analysis.py's
        # _record_fit_on_receiver/transform_transformer docstrings.
        self.fitted_from: tuple = fitted_from
        # Where the taint was introduced: {"cell", "line", "call", "variable"}
        # of the preprocessing call (fit_transform, or the fit() behind a
        # leaky transform()). Provenance only - deliberately left out of
        # __eq__ so it never affects fixpoint convergence.
        self.taint_origin: dict = taint_origin
        # Usage -> the same location shape, for the trainKB/testKB call that
        # added that usage (e.g. TRAIN -> `lr.fit(X, y)`). Also provenance
        # only, also outside __eq__.
        self.usage_sites: dict = dict(usage_sites or {})

    def apply_taint(self, new_source: str, origin: dict = None) -> DataLeakAbstractDomain:
        self.taint = True
        self.dfs.rename_sources(new_source)
        if origin:
            self.taint_origin = origin
        return self

    def __eq__(self, other: DataLeakAbstractDomain) -> bool:
        if self.dull == other.dull:
            return True

        return self.dfs == other.dfs and self.taint == other.taint and self.usages == other.usages

    def __ne__(self, other: DataLeakAbstractDomain) -> bool:
        return not self == other

class DataLeakAbstractState(AbstractState):
    def __init__(self) -> None:
        '''
        Object that maps variables to corresponding Abstract Domain element.
        '''
        self.state: dict[str, DataLeakAbstractDomain] = dict()
        super().__init__()
        

    def __getitem__(self, key: str) -> DataLeakAbstractDomain:
        return self.state[key]

    def condition(self, cell_IR: IntermediateRepresentations, node: Node, errors: list[ErrorInfo]) -> list:        
        test_domain: list[DataLeakAbstractDomain] = [e for e in self.state.values() if Usage.TEST in e.usages]
        train_domain: list[DataLeakAbstractDomain] = [e for e in self.state.values() if Usage.TRAIN in e.usages]

        for test_dfs, train_dfs in product(test_domain, train_domain):
            if test_dfs.dfs.cartesian_overlap(train_dfs.dfs):
                label, call, func = self._sink_of(node, cell_IR)
                if label is not None:
                    other = self._counterpart_site(cell_IR, node, train_dfs, test_dfs)
                    if other:
                        with_what = f"`{other['variable']}` used by `{other['call']}` {_where(other)}"
                    else:
                        with_what = "the model's training data"
                    message = (f"Train/test overlap: `{label}` passed to `{call}`{self._in(func)} shares rows with "
                               f"{with_what}, so evaluation scores are inflated.")
                    errors.append(ErrorInfo(cell_IR.cell_id, node.line_number, label, ErrorType.TERMINAL, message, source=other))
                return errors
            
            if test_dfs.dfs.cartesian_overlap(train_dfs.dfs, weak=True) and test_dfs.taint and train_dfs.taint:
                label, call, func = self._sink_of(node, cell_IR)
                if label is not None:
                    origin = test_dfs.taint_origin or train_dfs.taint_origin
                    if origin:
                        cause = (f"`{origin['call']}` {_where(origin)} ran on "
                                 f"unsplit `{origin['variable']}` before the train/test split")
                    else:
                        cause = "a preprocessing step ran on the data before the train/test split"
                    message = (f"Preprocessing leak: `{label}` passed to `{call}`{self._in(func)} is contaminated because "
                               f"{cause}, so test-set statistics leak into the model.")
                    errors.append(ErrorInfo(cell_IR.cell_id, node.line_number, label, ErrorType.TERMINAL, message, source=origin))
                return errors
        return errors

    @staticmethod
    def _counterpart_site(cell_IR, node: Node, train_dfs: DataLeakAbstractDomain, test_dfs: DataLeakAbstractDomain) -> dict:
        '''
        For an overlap leak, the location of the OTHER half of the train/test
        pair - the fit() call when the error fires at predict(), and vice
        versa. None if that site was never recorded.
        '''
        train_site = train_dfs.usage_sites.get(Usage.TRAIN)
        test_site = test_dfs.usage_sites.get(Usage.TEST)
        here = (cell_IR.cell_id, node.line_number)
        if train_site and (train_site["cell"], train_site["line"]) == here:
            return test_site
        return train_site

    @staticmethod
    def _in(func) -> str:
        return f" (in `{func}`)" if func else ""

    @staticmethod
    def _sink_of(node: Node, cell_IR=None) -> tuple:
        '''
        (variable, call name, inlined function or None) at the node completing
        the train/test pair, e.g. ("X_test", "lr.predict", None), with
        inlined-call names restored. Variable is None when the call's first
        argument isn't a plain name - no error is reported for that shape.
        '''
        label, call = None, None
        if not isinstance(node, AssignmentNode):
            label = call = node.label
        else:
            call = getattr(node, "func_name", None)
            if isinstance(node.ast_node, ast.Call):
                f = node.ast_node.func
                call = call or (f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None))
                if node.ast_node.args and isinstance(node.ast_node.args[0], ast.Name):
                    label = node.ast_node.args[0].id

        label, label_func = demangle(label, cell_IR)
        call, call_func = demangle(call, cell_IR)
        return label, call, label_func or call_func

    def aug_join(self, other: DataLeakAbstractState) -> DataLeakAbstractState:
        '''
        Augment join operation on two Abstract states.
        '''
        for other_var, other_domain in other.state.items():

            if other_var not in self.state:
                self.state[other_var] = other_domain
                continue

            self.state[other_var] = DataLeakAbstractDomain(
                self.state[other_var].dfs | other_domain.dfs,
                self.state[other_var].taint | other_domain.taint,
                self.state[other_var].usages.union(other_domain.usages),
                self.state[other_var].dull | other_domain.dull,
                self.state[other_var].fitted_from or other_domain.fitted_from,
                self.state[other_var].taint_origin or other_domain.taint_origin,
                {**other_domain.usage_sites, **self.state[other_var].usage_sites}
            )

        return self

    def contains(self, other: DataLeakAbstractState) -> bool:
        self_filtered = {k:v for k, v in self.state.items() if k[0] != '~'}
        other_filtered = {k:v for k, v in other.state.items() if k[0] != '~'}
        return other_filtered.items() <= self_filtered.items() 

    def projection(self) -> set[str]:
        '''
        Get all variables that map to Data Frame Sets with non-empty rows.
        TODO: Resolve variable binding for ones that do not interfere with dataleak domain (currently solved with dull)
        '''
        return {var for var, dom in self.state.items() if dom.dfs.some_rows() or dom.dull}

    def __eq__(self, other: DataLeakAbstractState) -> bool:
        return self.state == other.state

    def __ne__(self, other: DataLeakAbstractState) -> bool:
        return not self == other
