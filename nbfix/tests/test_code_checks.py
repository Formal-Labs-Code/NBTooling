import unittest

from nbfix.code_checks import (
    first_binding, fits_on_test_data, misaligned_splits, produced_names, readers, undefined_reads, unused_results,
)

SPLIT = "X_train, X_test, y_train, y_test = train_test_split(X, y)"


class TestFitsOnTestData(unittest.TestCase):
    def check(self, *cells):
        return fits_on_test_data(dict(enumerate(cells)))

    def test_fit_on_the_test_split(self):
        self.assertEqual(self.check(SPLIT, "scaler.fit(X_test)"), {("scaler.fit", "X_test")})

    def test_fit_on_data_derived_from_the_test_split(self):
        self.assertEqual(self.check(SPLIT, "X_t = X_test[cols]\nX_t2 = clean(X_t)", "sel.fit_transform(X_t2)"),
                         {("sel.fit_transform", "X_t2")})

    def test_local_function_fitting_its_argument(self):
        helper = "def prep(X, scale=True):\n    s = StandardScaler()\n    return s.fit_transform(X)"
        self.assertEqual(self.check(helper, SPLIT, "X_train = prep(X_train)\nX_test = prep(X_test)"),
                         {("prep", "X_test")})

    def test_fitting_on_train_and_transforming_test_is_fine(self):
        helper = "def prep(X, scaler):\n    return scaler.transform(X)"
        self.assertEqual(self.check(helper, SPLIT, "scaler.fit(X_train)", "X_test = prep(X_test, scaler)",
                                    "lr.fit(X_train, y_train)", "lr.score(X_test, y_test)"), set())


class TestMisalignedSplits(unittest.TestCase):
    def test_labels_resplit_separately_from_features(self):
        # The wrong fix qwen2.5-coder:14b gave for dataleak_two_leaks_true.py in run 5.
        cells = {0: SPLIT, 1: "A = scaler.fit_transform(X_train)", 2: "b_train, b_test = train_test_split(y)",
                 3: "model.fit(A, b_train)"}
        self.assertEqual(misaligned_splits(cells), {("model.fit", "A", "b_train")})

    def test_features_and_labels_from_the_same_split(self):
        cells = {0: SPLIT, 1: "A = scaler.fit_transform(X_train)", 2: "model.fit(A, y_train)",
                 3: "model.score(scaler.transform(X_test), y_test)"}
        self.assertEqual(misaligned_splits(cells), set())


class TestDataFlow(unittest.TestCase):
    def test_produced_names(self):
        self.assertEqual(produced_names("X_res, y_res = smote.fit_resample(X, y)"), ["X_res", "y_res"])
        self.assertEqual(produced_names("scaler.fit(X)"), ["scaler"])
        self.assertEqual(produced_names("return scaler.fit_transform(X)"), [])
        self.assertEqual(produced_names("X_train, X_test = train_test_split("), [])

    def test_readers_are_later_cells_only(self):
        cells = {0: "X = load()", 1: "X = clean(X)", 2: "X = scale(X)", 3: "split(X)", 4: "print(1)"}
        self.assertEqual(readers(cells, "X", after=2), [3])


class TestUndefinedReads(unittest.TestCase):
    def test_first_binding(self):
        self.assertEqual(first_binding({0: "a = 1", 1: "b, c = f()"}, "c"), 1)
        self.assertIsNone(first_binding({0: "a = 1"}, "z"))
        self.assertEqual(first_binding({0: "c = c + 1", 1: "c = 0"}, "c", after=0), 1)

    def test_read_before_a_later_cell_defines_it(self):
        self.assertEqual(undefined_reads({0: "a = b", 1: "b = 1"}), {"b": 0})

    def test_function_bodies_see_module_names_bound_anywhere(self):
        self.assertEqual(undefined_reads({0: "def f():\n    return later\nxs = [i for i in range(3)]",
                                          1: "later = 1"}), {})

    def test_function_reading_a_name_nothing_binds(self):
        # The wrong fix qwen2.5-coder:7b gave for dataleak_two_leaks_true.py in run 6.
        main = ("def main(X, y):\n    A, A_test, b, b_test = train_test_split(X, y)\n"
                "    second.fit(A, b_train)\n    g = lambda v: v + 1\n    import os\n    return os.sep")
        imports = "from sklearn.model_selection import train_test_split\nfrom models import Model\n"
        self.assertEqual(undefined_reads({0: imports + "second = Model()", 1: main}), {"b_train": 1})


class TestUnusedResults(unittest.TestCase):
    def test_result_never_read_afterwards(self):
        self.assertEqual(unused_results({0: "X_r, y_r = sm.fit_resample(X, y)", 1: "lr.fit(X, y)"}, {"fit_resample"}),
                         {"X_r", "y_r"})


if __name__ == "__main__":
    unittest.main()
