from collections import defaultdict
from ..parser.cfg_nodes import AssignmentCallNode, AssignmentNode, BBorBInode, RestoreNode
from .abs_states.dataleak_abs_state import (
    demangle,
    DataLeakAbstractDomain,
    DataLeakAbstractState,
    DataFrameSet,
    Usage
)
from .abs_domains.dataleak_lattice.data_frame import DataFrame, Rows, Columns
from ..ir.intermediate_representations import IntermediateRepresentations
from .runner.analysis_results import Result, ErrorType
from .runner.runners import Runner
from .runner.stats import Stats

from ..parser import ast_nodes as ast
import copy
from .analysis import Analysis

DATA_LEAK = "Data Leak Analysis"


class DataLeakAnalysis(Analysis):
    def __init__(self):
        super().__init__()
        self.resetKB = {'read_csv', 'genfromtxt', 'read_excel', 'read_parquet', 'read_sql', 'read_json', 'read_pickle', 'load', 'loadtxt'}
        self.taintKB = {'normalize', 'fit_transform', 'fit_resample'}
        self.trainKB = {'fit', 'train'}
        self.testKB = {'predict', 'predict_proba', 'score', 'decision_function', 'evaluate'}
        self.splitKB = {'train_test_split'}
        self.dropKB = {'drop'}
        # Separate from testKB deliberately: .transform() is called
        # routinely on BOTH train and test splits (X_train_scaled =
        # scaler.transform(X_train) is completely normal), so it can't be
        # treated as a "this is test data" signal the way .predict() is -
        # see transform_transformer's docstring for what it actually does.
        self.transformKB = {'transform'}
        self.necessary = set[int]()
        self.abstract_state = DataLeakAbstractState()


    def F_transformer(self, cfg_node, a_state: DataLeakAbstractState, cell_IR):
        '''
        Gluing function for various construct transformers. 
        '''
        if isinstance(cfg_node, RestoreNode):
            other: DataLeakAbstractDomain = a_state.state.get(cfg_node.right_hand_side[0])
            if other is None:
                a_state.state[cfg_node.left_hand_side] = DataLeakAbstractDomain(dull=True)
                return a_state
            a_state.state[cfg_node.left_hand_side] = DataLeakAbstractDomain(copy.deepcopy(other.dfs), other.taint, other.usages.copy(), dull=other.dull, taint_origin=other.taint_origin, usage_sites=other.usage_sites, tainted_sources=other.tainted_sources)

        if not cfg_node.ast_node:
            return a_state

        # as_transformed: DataLeakAbstractState = copy.deepcopy(a_state)
        as_transformed = a_state

        if isinstance(cfg_node, AssignmentCallNode):
            '''
            Renames intermediate call nodes and transfers them to variables
            '''
            lhs, rhs = cfg_node.left_hand_side, cfg_node.right_hand_side[0]

            if rhs not in as_transformed.state:
                return as_transformed

            as_transformed.state[lhs] = as_transformed.state[rhs]
            as_transformed.state.pop(rhs)

            return as_transformed

        if isinstance(cfg_node, BBorBInode):
            '''
            Black-box or Built-in functions include majority of data science library functions.
            '''
            func_name = self._extract_func(cfg_node)

            if func_name in self.splitKB:
                self.split_transformer(cfg_node, as_transformed)

            elif func_name in self.dropKB:
                self.drop_transformer(cfg_node, as_transformed)
                
            elif func_name in self.resetKB:
                self.reset_transformer(cfg_node, as_transformed)

            elif func_name in self.taintKB:
                self.taint_transformer(cfg_node, as_transformed, cell_IR)

            elif func_name in self.trainKB:
                # Should be visitor ideally.
                x_kw = self._extract_kw(cfg_node, 'x')
                x = x_kw[0].value.id if x_kw else None
                self._tag_usage(self._clone_assign(as_transformed, cfg_node, override_source=x), Usage.TRAIN, cfg_node, cell_IR, x)
                self._record_fit_on_receiver(cfg_node, as_transformed, cell_IR)

            elif func_name in self.testKB:
                self._tag_usage(self._clone_assign(as_transformed, cfg_node), Usage.TEST, cfg_node, cell_IR)

            elif func_name in self.transformKB:
                self.transform_transformer(cfg_node, as_transformed, cell_IR)

            else:
                # All other function calls not related to Data Leak.
                as_transformed.state[cfg_node.left_hand_side] = DataLeakAbstractDomain(dull=True)

            return as_transformed

        if isinstance(cfg_node, AssignmentNode):

            if not hasattr(cfg_node.ast_node, "value"):
                # `with ... as name`: name is bound to whatever __enter__
                # returns, which says nothing about the data - opaque.
                as_transformed.state[cfg_node.left_hand_side] = DataLeakAbstractDomain(dull=True)

            elif isinstance(cfg_node.ast_node.value, ast.Name):
                '''
                Plain variable to variable assignment.
                '''
                self._clone_assign(as_transformed, cfg_node, override_source=cfg_node.ast_node.value.id)

            elif isinstance(cfg_node.ast_node.value, ast.Subscript):

                if isinstance(cfg_node.ast_node.value.value, ast.Attribute) and isinstance(cfg_node.ast_node.value.value.value, ast.Name):
                    self.iloc_transformer(cfg_node, as_transformed)

                elif isinstance(cfg_node.ast_node.value.value, ast.Name):
                    self.slice_transformer(cfg_node, as_transformed)

            else:
                as_transformed.state[cfg_node.left_hand_side] = DataLeakAbstractDomain(dull=True)

            return as_transformed

        return as_transformed


    def split_transformer(self, cfg_node, as_transformed):
        '''
        Dealing with train_test_split return values. For simplicity, always truncate based on test size.
        train_test_split returns a (train, test) pair per array argument, and the CFG gives each
        unpacking target its own call node tagged with its position (target_index): target i is
        the train (even i) or test (odd i) part of argument i // 2. Position-based rather than a
        visit counter, so revisiting a node during the fixpoint doesn't shift the assignment.
        '''
        test_size_kw, train_size_kw = self._extract_kw(cfg_node, 'test_size'), self._extract_kw(cfg_node, 'train_size')
        train_size = train_size_kw[0].value.value if (train_size_kw and isinstance(train_size_kw[0].value, ast.Constant)) else None
        test_size = test_size_kw[0].value.value if (test_size_kw and isinstance(test_size_kw[0].value, ast.Constant)) else (0.25 if not train_size else None)

        test_size = round(1.0 - train_size, 4) if not test_size else test_size
        
        index = getattr(cfg_node, "target_index", 0)
        arg_pos = index // 2
        invert_trunc = index % 2 == 1

        self._clone_assign(as_transformed, cfg_node, arg_pos= arg_pos)
        as_transformed.state[cfg_node.left_hand_side].dfs.truncate_rows(test_size, inverse=invert_trunc)


    def drop_transformer(self, cfg_node, as_transformed):
        convert = {'columns': 1, 'rows' : 1}
        axis_kw, columns_kw = self._extract_kw(cfg_node, 'axis'), self._extract_kw(cfg_node, 'columns')
        index_kw, labels_kw = self._extract_kw(cfg_node, 'index'), self._extract_kw(cfg_node, 'labels')

        if columns_kw:
            args, axis = columns_kw[0].value, 1

        elif index_kw: 
            args, axis =  index_kw[0].value, 0

        else:                
            args = labels_kw[0].value if labels_kw else cfg_node.ast_node.args[0]
            if axis_kw:
                axis = axis_kw[0].value.value
            elif len(cfg_node.ast_node.args) >= 2:
                axis = cfg_node.ast_node.args[1].value
            else:
                axis = 0
                
        if isinstance(args, ast.Attribute) or isinstance(args, ast.Name) or (isinstance(args, ast.List) and isinstance(args.elts[0], ast.Name)):
            '''
            Column arrays taken as an ast.Name. Requires another analysis to be tracked.
            '''
            return
            
        self._clone_assign(as_transformed, cfg_node, override_source=cfg_node.ast_node.func.value.id)
        axis = convert[axis] if isinstance(axis, str) else axis
        
        
        if isinstance(args, ast.List):
            if axis == 0 and args.elts and isinstance(args.elts[0].value, str):
                return
            for arg in args.elts:
                self._invoke_drop(arg.value, axis, as_transformed.state[cfg_node.left_hand_side])
        else:
            if axis == 0 and isinstance(args.value, str):
                return
            self._invoke_drop(args.value, axis, as_transformed.state[cfg_node.left_hand_side])


    def reset_transformer(self, cfg_node, as_transformed):
        '''
        If argument is local directory plus file name track only the file name.
        '''
        source_arg = self._extract_filename(cfg_node)
        
        df = DataFrame(source_arg, Columns(all=True), Rows())
        dfs = DataFrameSet({source_arg: [df]})

        as_transformed.state[cfg_node.left_hand_side] = DataLeakAbstractDomain(dfs, False, set[Usage]())


    def taint_transformer(self, cfg_node, as_transformed, cell_IR):
        '''
        Handles of variations of an argument (first positional).
        E.g. plain dataset name, attribute like df.values etc...
        '''
        x_kw = self._extract_kw(cfg_node, 'X')
        arg = x_kw[0].value.id if x_kw else cfg_node.ast_node.args[0]

        if isinstance(arg, ast.Name):
            new_source = arg.id

        elif isinstance(arg, ast.Subscript):
            new_source = arg.value.value.id if isinstance(arg.value, ast.Attribute) else arg.value.id
            
        elif isinstance(arg, ast.Call):
            '''
            Temporary
            '''
            if isinstance(arg.func.value, ast.Name):
                new_source = arg.func.value.id
            else:
                new_source = arg.func.value.value.value.id
        
        elif isinstance(arg, ast.Attribute):
            new_source = arg.value.id
                
            
        origin = self._origin(cfg_node, cell_IR, new_source)
        self._clone_assign(as_transformed, cfg_node, override_source= new_source).apply_taint(new_source= new_source, origin=origin)


    def _origin(self, cfg_node, cell_IR, variable) -> dict:
        '''
        Location of a taint-introducing call, reported as a finding's source.
        '''
        call, call_func = demangle(cfg_node.func_name, cell_IR)
        variable, var_func = demangle(variable, cell_IR)
        origin = {"cell": cell_IR.cell_id, "line": cfg_node.line_number, "call": call, "variable": variable}
        if call_func or var_func:
            origin["function"] = call_func or var_func
        return origin


    def _tag_usage(self, domain: DataLeakAbstractDomain, usage: Usage, cfg_node, cell_IR, variable=None) -> None:
        '''
        Marks the call's result as used for training/testing and records
        where, so overlap findings can point at the other half of the pair.
        '''
        domain.usages.add(usage)
        if variable is None and cfg_node.ast_node.args and isinstance(cfg_node.ast_node.args[0], ast.Name):
            variable = cfg_node.ast_node.args[0].id
        domain.usage_sites[usage] = self._origin(cfg_node, cell_IR, variable)


    def _record_fit_on_receiver(self, cfg_node, as_transformed, cell_IR) -> None:
        '''
        When a trainKB call (.fit()/.train()) is a method call on a
        receiver object - e.g. `scaler.fit(X)`, as opposed to a bare/
        module-level call - records on the RECEIVER's own state entry
        (as_transformed.state["scaler"], not the fit call's assigned
        result) which data source it was fit with, and whether that
        source's rows were still the full, untouched range at fit time
        (i.e. this fit happened before ANY split ever touched it).

        transform_transformer consults this later to distinguish the real
        leak (fit on the whole dataset, transform() separately per split)
        from the correct, recommended pattern (fit on an already-split
        subset, transform() both splits) - both are the same call shape,
        so this is the actual distinguishing signal, not something a
        keyword-set addition alone could express. See that method's
        docstring for the full mechanism.

        Silently does nothing if the shape doesn't match what's needed
        (not a method call on a plain-Name receiver, argument isn't a
        plain tracked variable) - this is a best-effort enhancement on
        top of the existing Usage.TRAIN tagging above, never required for
        that to keep working, and never raises.
        '''
        func = cfg_node.ast_node.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
            return
        receiver = func.value.id

        if not (cfg_node.ast_node.args and isinstance(cfg_node.ast_node.args[0], ast.Name)):
            return
        arg_name = cfg_node.ast_node.args[0].id

        arg_domain = as_transformed.state.get(arg_name)
        if arg_domain is None or arg_domain.dull:
            return

        was_full = bool(arg_domain.dfs.frames) and all(
            df.rows.is_full() for l_df in arg_domain.dfs.frames.values() for df in l_df
        )

        receiver_domain = as_transformed.state.get(receiver)
        if receiver_domain is None:
            receiver_domain = DataLeakAbstractDomain(dull=True)
            as_transformed.state[receiver] = receiver_domain
        receiver_domain.fitted_from = (arg_name, was_full, self._origin(cfg_node, cell_IR, arg_name))


    def transform_transformer(self, cfg_node, as_transformed, cell_IR) -> None:
        '''
        Handles `receiver.transform(arg)`.

        If `receiver` was previously recorded (via _record_fit_on_receiver)
        as fit on a source that was still the FULL, unsplit range at fit
        time, this is the real leak pattern - fit on the whole dataset,
        then transform() called separately per split - so the result is
        tainted using the ORIGINAL fit source (not this call's own
        argument). That makes every split's .transform() output share one
        tainted source, the same shape fit_transform-before-split already
        produces, so DataLeakAbstractState.condition()'s existing weak-
        overlap+taint check catches it once those results reach a
        trainKB/testKB call downstream - no changes needed there.

        Otherwise (receiver was fit on already-split/truncated rows - the
        correct, recommended pattern - or was never recorded as fit()-ed
        at all) falls back to taint_transformer's existing behavior:
        taint from THIS call's own argument, same as fit_transform. That
        keeps every split's .transform() output on its OWN distinct
        source, so no false leak - verified directly against a fixture of
        exactly this pattern (fit on train only, transform both splits)
        before trusting it, not assumed.
        '''
        func = cfg_node.ast_node.func
        receiver = func.value.id if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) else None
        receiver_domain = as_transformed.state.get(receiver) if receiver else None

        if receiver_domain and receiver_domain.fitted_from and receiver_domain.fitted_from[1]:
            # Rows come from THIS call's argument (each split keeps its own
            # range); only the source name is the shared fit source.
            new_source, _, fit_origin = receiver_domain.fitted_from
            self._clone_assign(as_transformed, cfg_node).apply_taint(new_source=new_source, origin=fit_origin)
            return

        self.taint_transformer(cfg_node, as_transformed, cell_IR)


    def iloc_transformer(self, cfg_node, as_transformed):
        '''
        'Iloc-ed' variable to variable assignment.
        '''
        to_slice: DataLeakAbstractDomain = self._clone_assign(as_transformed, cfg_node, override_source=cfg_node.ast_node.value.value.value.id)
        slice = cfg_node.ast_node.value.slice

        if not isinstance(slice, ast.Slice) or self._incalculable(slice):
            return
            
        lw, up = self._get_if_int(slice.lower), self._get_if_int(slice.upper)                        
        to_slice.dfs.slice_rows(lw, up)


    def slice_transformer(self, cfg_node, as_transformed):
        '''
        Handles all variations of slicing. Row picking by ranges or column picking by names.
        '''
        to_slice: DataLeakAbstractDomain = self._clone_assign(as_transformed, cfg_node, override_source=cfg_node.ast_node.value.value.id)

        slice = cfg_node.ast_node.value.slice

        columns = list[str]()
        if isinstance(slice, ast.List):
            columns = {const.value : True for const in slice.elts}

        elif isinstance(slice, ast.Constant):
            columns = {slice.value: True}

        elif isinstance(slice, ast.Slice) and not self._incalculable(slice):
            '''
            ast.Slice -> Picks rows from certain range.
            '''

            lw, up = self._get_if_int(slice.lower), self._get_if_int(slice.upper)             
            to_slice.dfs.slice_rows(lw, up)

            return        

        else:
            '''
            Other possibilities for a slice field:
            
            ast.Compare -> Picks rows with certain numerical values. Statically cannot be calculated.
            Takes the worst case scenario, all rows.

            ast.Name -> Picks columns from provided list(named).
            Discuss implementing column only tracking system for precision improvement.
            '''    
            return

        to_slice.dfs.pick_columns(columns)


    def _extract_func(self, cfg_node):
        '''
        Extracts function name depending on its invocation form: as plain name or attribute of other name.
        '''
        func = cfg_node.ast_node.func
        return func.id if isinstance(cfg_node.ast_node.func, ast.Name) else func.attr 


    def _extract_kw(self, cfg_node, name):
        return list(filter(lambda kw: kw.arg == name, cfg_node.ast_node.keywords))


    def _extract_filename(self, cfg_node):
        if isinstance(cfg_node.ast_node.args[0], ast.BinOp):
            return cfg_node.ast_node.args[0].right.value

        elif isinstance(cfg_node.ast_node.args[0], ast.Constant):
            return cfg_node.ast_node.args[0].value
    
        else:
            # Consume the whole node as a source for now.
            return ast.dump(cfg_node.ast_node.args[0])  

    def _invoke_drop(self, value, axis, domain : DataLeakAbstractDomain) -> None:
        '''
        Drop wrapper for readability.
        '''
        if axis == 1:
            domain.dfs.drop_columns(value)
        else:
            domain.dfs.drop_rows(int(value))

    def _get_if_int(self, node):
        '''
        Covers situations where negative numbers are represented with ast.UnaryOp enveloping ast.Constant.
        Returns None if illegal.
        '''
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.UnaryOp):
            return - node.operand.value
        return None

    def _incalculable(self, slice: ast.Slice):
        return (slice.lower != None and not self._get_if_int(slice.lower)) or (slice.upper != None and not self._get_if_int(slice.upper))

    def _clone_assign(self,as_transformed: DataLeakAbstractState, cfg_node, override_source=None, arg_pos=0) -> DataLeakAbstractDomain:
        '''
        Helper function that extracts source and lhs from cfg node and performs needed deepcopy and assignment.
        '''
        if override_source:
            source_arg = override_source

        else:
            '''
            Default way of source extraction.
            Try to handle in future: 
                Argument not data frame but function that returns one.
            '''
            if arg_pos >= len(cfg_node.ast_node.args) or not isinstance(cfg_node.ast_node.args[arg_pos], ast.Name):
                as_transformed.state[cfg_node.left_hand_side] = DataLeakAbstractDomain(dull= True)
                return as_transformed.state[cfg_node.left_hand_side]

            source_arg = cfg_node.ast_node.args[arg_pos].id

        other: DataLeakAbstractDomain = as_transformed.state.get(source_arg)
        if other is None:
            # Source never reached this path's state (e.g. a cell explored
            # out of order) - treat like an untracked argument, don't crash.
            as_transformed.state[cfg_node.left_hand_side] = DataLeakAbstractDomain(dull= True)
            return as_transformed.state[cfg_node.left_hand_side]

        as_transformed.state[cfg_node.left_hand_side] = DataLeakAbstractDomain(copy.deepcopy(other.dfs), other.taint, other.usages.copy(), dull=other.dull, taint_origin=other.taint_origin, usage_sites=other.usage_sites, tainted_sources=other.tainted_sources)

        return as_transformed.state[cfg_node.left_hand_side]

    def phi_condition(self, current: set, pre: set, cell_IR: IntermediateRepresentations):
        return (pre <= current)

    def update_abstract_state(self, cell_IR, notebook_IR):
        # The cell ran to completion, leak or not: stopping at the leak would
        # leave the kernel state half-updated (a call's result never moved
        # onto its variable), and that stale state then reports leaks against
        # later, fixed code.
        self.abstract_state, _ = Runner(self.stats, defaultdict(DataLeakAbstractState), notebook_IR).intra_fixpoint_runner(
            cell_IR, self, self.abstract_state, stop_at_terminal=False)

    def combine_states(self, states: list[DataLeakAbstractState]):
        if len(states) == 1:
            return states[0]

        res_state = DataLeakAbstractState()
        for s in states:
            res_state.aug_join(s)

        return res_state

    def find_necessary_cells(self, notebook_IR):

        dataleak_kb : set[str] = (self.testKB | self.trainKB)

        for cell, IR in notebook_IR.items():
            for node in IR.CFG.nodes:
                if isinstance(node, BBorBInode):
                    if self._extract_func(node) in dataleak_kb:
                        self.necessary.add(cell)
                        break

                elif isinstance(node, AssignmentNode):
                    '''
                    Covers both AssignmentCallNode and AssignmentNode
                    '''
                    self.necessary.add(cell)
                    break

    def analyze_notebook(self, notebook_IR, old_cell_IR = None, level = 4, filename=""):
        
        self.notebook_IR: dict[int, IntermediateRepresentations] = notebook_IR

        pre = notebook_IR[old_cell_IR.cell_id].UDA.def_use_chains.unbound_names - notebook_IR[old_cell_IR.cell_id].UDA.funcs
        if not self.phi_condition(self.abstract_state.projection(), pre, old_cell_IR):
            return Result()

        stat = Stats(old_cell_IR.cell_id, filename) 
        stat.log_start()

        result = Runner(stat, defaultdict(DataLeakAbstractState), notebook_IR).inter_fixpoint_runner(
            self,
            old_cell_IR.cell_id,
            abstract_state=self.abstract_state,
            K=level,
            cpath=[],
            results = Result()
        )

        stat.log_end()
        self.stats.append(stat)
        return result

    def summarize_result(self, result: Result) -> Result:
        summarized_result: Result = result.distinct_errors()
        for path_result in result.distinct_errors().path_results:
            for err in path_result.error_infos:
                if err.error_type == ErrorType.TERMINAL:
                    err.error_message += " Path that will lead to this: " + str(path_result.path)
        
        return summarized_result

    def calculate_pre(self, cell_IR):
        return cell_IR.UDA.unbound_final
