from ast import Assert
import unittest

from nbcore.ir.intermediate_representations import IntermediateRepresentations
from nbcore.analyses.abs_states.dataleak_abs_state import DataLeakAbstractDomain, DataLeakAbstractState, Usage
from nbcore.analyses.abs_domains.dataleak_lattice.data_frame_sets import DataFrameSet
from nbcore.analyses.abs_domains.dataleak_lattice.data_frame import DataFrame
from nbcore.analyses.abs_domains.dataleak_lattice.columns import Columns
from nbcore.analyses.abs_domains.dataleak_lattice.rows import Rows
from nbcore.parser.cfg_nodes import Node
from nbcore.analyses.runner.analysis_results import ErrorInfo, ErrorType

class TestDataFrames(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        return super().setUp()

    def test_df_join(self):
        d1: DataFrame = DataFrame(
            'df',
            Columns({'kolona11': True, 'kolona12': False, 'kolona13': True}),
            Rows(8, 15),
        )

        d2: DataFrame = DataFrame(
            'df',
            Columns({'kolona21': True, 'kolona22': False, 'kolona13': True}),
            Rows(10, 22),
        )

        self.assertEqual(
            d1 | d2,
            DataFrame(
                'df',
                Columns(
                    {
                        'kolona11': True,
                        'kolona12': False,
                        'kolona13': True,
                        'kolona21': True,
                        'kolona22': False,
                    }
                ),
                Rows(8, 22),
            ),
        )

        # Test reduction operator
        d1.columns.columns['kolona13'] = False
        self.assertEqual(
            d1 | d2,
            DataFrame(
                'df',
                Columns(
                    {
                        'kolona11': True,
                        'kolona12': False,
                        'kolona21': True,
                        'kolona22': False,
                    }
                ),
                Rows(8, 22),
            ),
        )

    def test_df_meet(self):
        d1: DataFrame = DataFrame(
            'df',
            Columns({'kolona11': True, 'kolona22': False, 'kolona13': True}),
            Rows(10, 15),
        )

        d2: DataFrame = DataFrame(
            'df',
            Columns({'kolona11': True, 'kolona22': True, 'kolona23': True}),
            Rows(13, 22),
        )
        self.assertEqual(
            d1 & d2,
            DataFrame('df', Columns({'kolona11': True}), Rows(13, 15)),
        )

    def test_df_overlap(self):
        d1: DataFrame = DataFrame(
            'df',
            Columns({'kolona11': True, 'kolona22': False, 'kolona13': True}),
            Rows(10, 15),
        )

        d2: DataFrame = DataFrame(
            'df',
            Columns({'kolona11': True, 'kolona22': True, 'kolona23': True}),
            Rows(13, 22),
        )

        self.assertEqual(True, d1.overlap(d2))

        d2: DataFrame = DataFrame(
            'df',
            Columns({'kolona11': False, 'kolona22': True, 'kolona23': False}),
            Rows(13, 22),
        )

        self.assertEqual(False, d1.overlap(d2))

        d2: DataFrame = DataFrame(
            'df',
            Columns({'kolona11': True, 'kolona22': True, 'kolona23': True}),
            Rows(4, 8),
        )

        self.assertEqual(False, d1.overlap(d2))

    def test_df_subtraction(self):
        d1: DataFrame = DataFrame(
            'df',
            Columns({'kolona11': True, 'kolona22': False, 'kolona13': True}),
            Rows(10, 15),
        )

        d2: DataFrame = DataFrame(
            'df',
            Columns({'kolona11': True, 'kolona22': True, 'kolona23': True}),
            Rows(13, 22),
        )

        d_ref: DataFrame = DataFrame(
            'df', Columns({'kolona13': True, 'kolona22': False}), Rows(10, 12)
        )
        self.assertEqual(d1 - d2, {d_ref})

        d2.rows = Rows(15, 20)
        d_ref.rows = Rows(10, 14)
        self.assertEqual(d1 - d2, {d_ref})

        d2.rows = Rows(6, 12)
        d_ref.rows = Rows(13, 15)
        self.assertEqual(d1 - d2, {d_ref})

        d2.rows = Rows(10, 10)
        d_ref.rows = Rows(11, 15)
        self.assertEqual(d1 - d2, {d_ref})

        d2.rows = Rows(15, 15)
        d_ref.rows = Rows(10, 14)
        self.assertEqual(d1 - d2, {d_ref})

        d2.rows = Rows(10, 15)
        d_ref.rows = Rows(-1, -1)
        self.assertEqual(d1 - d2, {d_ref})

        d2.rows = Rows(10, 11)
        d_ref.rows = Rows(12, 15)
        self.assertEqual(d1 - d2, {d_ref})

        d2.rows = Rows(14, 15)
        d_ref.rows = Rows(10, 13)
        self.assertEqual(d1 - d2, {d_ref})

        d2.rows = Rows(12, 13)
        d_ref.rows = Rows(10, 11)
        d_ref2: DataFrame = DataFrame(
            'df', Columns({'kolona13': True, 'kolona22': False}), Rows(14, 15)
        )
        self.assertEqual(d1 - d2, {d_ref, d_ref2})

        d2.columns = Columns(
            {'kolona11': True, 'kolona22': False, 'kolona13': True, 'kolona33': False}
        )
        d_ref.columns = d_ref2.columns = Columns({})

        self.assertEqual(d1 - d2, {d_ref, d_ref2})

    def test_data_frame_sets(self):
        dfs1: DataFrameSet = DataFrameSet(
            {
                'file1': [DataFrame('file1', Columns({'id': True}), Rows(1, 10))],
                'file2': [DataFrame('file2', Columns({'name': True}), Rows(0, 100))],
            }
        )
        dfs2 = DataFrameSet(
            {
            'file1': [DataFrame('file1', Columns({'id': True}), Rows(9, 12))],
            'file3': [DataFrame('file3', Columns({'zip': True}), Rows(0, 100))],
            }
        )

        ref_dfs = DataFrameSet(
            {
            'file1': [DataFrame('file1', Columns({'id': True}), Rows(1, 12))],
            'file2': [DataFrame('file2', Columns({'name': True}), Rows(0, 100))],
            'file3': [DataFrame('file3', Columns({'zip': True}), Rows(0, 100))],
            }
        )

    
        self.assertEqual(dfs1 | dfs2, ref_dfs)

        dfs1.frames['file1'][0].rows = Rows(1, 6)


        ref_dfs = DataFrameSet(
            {
            'file1': [DataFrame('file1', Columns({'id': True}), Rows(1, 6)),
                    DataFrame('file1', Columns({'id': True}), Rows(9, 12))],
            'file2': [DataFrame('file2', Columns({'name': True}), Rows(0, 100))],
            'file3': [DataFrame('file3', Columns({'zip': True}), Rows(0, 100))],
            }
        )

        self.assertEqual(dfs1 | dfs2, ref_dfs)

    def test_condition_detection(self):

        dfs1: DataFrameSet = DataFrameSet({
            'x_selected': [DataFrame('x_selected', Columns({'col1': True, 'col2': True}), Rows(0, 2))]
        })

        dfs2: DataFrameSet = DataFrameSet({
            'x_selected': [DataFrame('x_selected', Columns({'col1': True, 'col2': True}), Rows(4, 5))]
        })        

        a_state: DataLeakAbstractState = DataLeakAbstractState()

        origin = {"cell": 0, "line": 3, "call": "scaler.fit_transform", "variable": "x_selected"}
        a_state.state['X_train'] = DataLeakAbstractDomain(dfs1, True, {Usage.TRAIN})
        a_state.state['X_test'] = DataLeakAbstractDomain(dfs2, True, {Usage.TEST}, taint_origin=origin)
        errors = []
        a_state.condition(IntermediateRepresentations("", 1), Node("", None, line_number=1, path=1), errors)
        self.assertEqual(errors, [ErrorInfo(1, 1, "", ErrorType.TERMINAL,
            "Preprocessing leak: `` passed to `` is contaminated because `scaler.fit_transform` (cell 0, line 3) "
            "ran on unsplit `x_selected` before the train/test split, so test-set statistics leak into the model.",
            source=origin)])

        a_state.state['X_test'] = DataLeakAbstractDomain(dfs2, False, {Usage.TEST})
        errors = []
        a_state.condition(IntermediateRepresentations("", 1), Node("", None, line_number=1, path=1), errors)
        self.assertEqual(errors, [])

        dfs2.frames['x_selected'][0].rows = Rows(2, 4)

        a_state.state['X_train'] = DataLeakAbstractDomain(dfs1, False, {Usage.TRAIN})
        a_state.state['X_test'] = DataLeakAbstractDomain(dfs2, True, {Usage.TEST})

        errors = []
        a_state.condition(IntermediateRepresentations("", 1), Node("", None, line_number=1, path=1), errors)
        self.assertEqual(errors, [ErrorInfo(1, 1, "", ErrorType.TERMINAL,
            "Train/test overlap: `` passed to `` shares rows with the model's training data, so evaluation scores are inflated.")])

        dfs1.frames['x_selected'][0].source = 'x_selected2'
        errors = []
        a_state.condition(IntermediateRepresentations("", 1), Node("", None, line_number=1, path=1), errors)
        self.assertEqual(errors, [])

    def test_overlap_source_is_counterpart_usage_site(self):
        dfs_train = DataFrameSet({'x': [DataFrame('x', Columns({'col1': True}), Rows(0, 4))]})
        dfs_test = DataFrameSet({'x': [DataFrame('x', Columns({'col1': True}), Rows(2, 6))]})
        fit_site = {"cell": 2, "line": 3, "call": "lr.fit", "variable": "X"}
        predict_site = {"cell": 3, "line": 1, "call": "lr.predict", "variable": "X_test"}

        a_state = DataLeakAbstractState()
        a_state.state['fit'] = DataLeakAbstractDomain(dfs_train, False, {Usage.TRAIN}, usage_sites={Usage.TRAIN: fit_site})
        a_state.state['pred'] = DataLeakAbstractDomain(dfs_test, False, {Usage.TEST}, usage_sites={Usage.TEST: predict_site})

        # Firing at predict() points back at fit() ...
        errors = []
        a_state.condition(IntermediateRepresentations("", 3), Node("", None, line_number=1, path=1), errors)
        self.assertEqual(errors[0].source, fit_site)

        # ... and firing at fit() (predict ran first) points at predict().
        errors = []
        a_state.condition(IntermediateRepresentations("", 2), Node("", None, line_number=3, path=1), errors)
        self.assertEqual(errors[0].source, predict_site)

    def test_non_dull_domains_with_different_taint_are_not_equal(self):
        dfs = DataFrameSet({'x': [DataFrame('x', Columns({'c': True}), Rows(0, 4))]})
        self.assertNotEqual(DataLeakAbstractDomain(dfs, False, set()), DataLeakAbstractDomain(dfs, True, set()))
        self.assertEqual(DataLeakAbstractDomain(dull=True), DataLeakAbstractDomain(dull=True))

    def test_join_keeps_taint_on_the_tainted_source_only(self):
        # `return X` (untainted) joined with `return scaler.fit_transform(X)`:
        # train/test splits of the untainted rows must not count as tainted.
        def domain(source, taint, rows, usage):
            dfs = DataFrameSet({source: [DataFrame(source, Columns({'c': True}), rows)]})
            return DataLeakAbstractDomain(dfs, taint, {usage})
        train = DataLeakAbstractState(); train.state['a'] = domain('raw', False, Rows(0, 4), Usage.TRAIN)
        other = DataLeakAbstractState(); other.state['a'] = domain('scaled_train', True, Rows(0, 4), Usage.TRAIN)
        train.aug_join(other)
        self.assertEqual(train.state['a'].tainted_sources, {'scaled_train'})

        a_state = DataLeakAbstractState()
        a_state.state['fit'] = train.state['a']
        a_state.state['pred'] = domain('raw', False, Rows(5, 8), Usage.TEST)
        a_state.state['pred'].taint = True  # tainted, but via a different source than 'raw'
        a_state.state['pred'].tainted_sources = {'scaled_test'}
        errors = []
        a_state.condition(IntermediateRepresentations("", 1), Node("", None, line_number=1, path=1), errors)
        self.assertEqual(errors, [])

    def test_demangle_restores_inlined_names_and_leaves_others(self):
        from types import SimpleNamespace
        from nbcore.analyses.abs_states.dataleak_abs_state import demangle
        cell_IR = SimpleNamespace(CFG=SimpleNamespace(inline_names={
            "~inline_0_test_model_model": ("model", "test_model"),
        }))
        self.assertEqual(demangle("~inline_0_test_model_model.predict", cell_IR), ("model.predict", "test_model"))
        self.assertEqual(demangle("lr.predict", cell_IR), ("lr.predict", None))
        # unknown mangled-looking names are left alone rather than guessed at
        self.assertEqual(demangle("~inline_9_g_x", cell_IR), ("~inline_9_g_x", None))
        self.assertEqual(demangle(None, cell_IR), (None, None))

    def test_clone_assign_with_unknown_source_is_dull_not_keyerror(self):
        from types import SimpleNamespace
        from nbcore.analyses.dataleak_analysis import DataLeakAnalysis
        node = SimpleNamespace(left_hand_side='copy', ast_node=None)
        a_state = DataLeakAbstractState()
        result = DataLeakAnalysis()._clone_assign(a_state, node, override_source='never_defined')
        self.assertTrue(result.dull)
        self.assertIs(a_state.state['copy'], result)

if __name__ == "__main__":
    unittest.main()
