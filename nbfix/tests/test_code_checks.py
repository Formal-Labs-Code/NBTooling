import unittest

from nbfix.code_checks import fits_on_test_data, undefined_reads, unused_results

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


class TestUndefinedReads(unittest.TestCase):
    def test_read_before_a_later_cell_defines_it(self):
        self.assertEqual(undefined_reads({0: "a = b", 1: "b = 1"}), {"b": 0})

    def test_function_bodies_and_comprehensions_are_not_reads(self):
        self.assertEqual(undefined_reads({0: "def f():\n    return later\nxs = [i for i in range(3)]"}), {})


class TestUnusedResults(unittest.TestCase):
    def test_result_never_read_afterwards(self):
        self.assertEqual(unused_results({0: "X_r, y_r = sm.fit_resample(X, y)", 1: "lr.fit(X, y)"}, {"fit_resample"}),
                         {"X_r", "y_r"})


if __name__ == "__main__":
    unittest.main()
