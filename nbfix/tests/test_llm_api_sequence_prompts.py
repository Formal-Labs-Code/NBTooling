import unittest

from nbfix.llm.api_sequence_context_builder import ApiSequenceContext
from nbfix.llm.api_sequence_prompts import API_SEQUENCE_SYSTEM_PROMPT, build_api_sequence_user_prompt


class TestBuildApiSequenceUserPrompt(unittest.TestCase):
    def test_includes_every_cells_current_code(self):
        context = ApiSequenceContext(
            cells={0: "model = M()", 1: "model.predict(X)"}, not_yet_run=set(), considered_cells={0, 1}
        )
        prompt = build_api_sequence_user_prompt(context)
        self.assertIn("Cell 0", prompt)
        self.assertIn("model = M()", prompt)
        self.assertIn("Cell 1", prompt)
        self.assertIn("model.predict(X)", prompt)

    def test_marks_not_yet_run_cells(self):
        context = ApiSequenceContext(
            cells={0: "model.fit(X, y)", 1: "model.predict(X)"},
            not_yet_run={0},
            considered_cells={0, 1},
        )
        prompt = build_api_sequence_user_prompt(context)
        self.assertIn("Cell 0  (not yet executed)", prompt)
        self.assertNotIn("Cell 1  (not yet executed)", prompt)


class TestApiSequenceSystemPrompt(unittest.TestCase):
    def test_documents_the_findings_schema(self):
        self.assertIn("findings", API_SEQUENCE_SYSTEM_PROMPT)
        self.assertIn("cell_id", API_SEQUENCE_SYSTEM_PROMPT)
        self.assertIn("message", API_SEQUENCE_SYSTEM_PROMPT)

    def test_does_not_name_specific_libraries(self):
        """
        Regression guard for the deliberate design choice this module's
        docstring explains: measured at 4/4 on tests/resources/llm_bench/
        api_sequence with qwen2.5-coder:14b using only "use your own
        knowledge," no enumerated library-specific patterns. Naming an
        actual library here would defeat the point of testing whether
        the LLM's own pretrained knowledge is sufficient.

        Narrower than the original version of this test, which also
        blocked generic method names like "fit("/"transform(" - dropped
        after confirming (not assumed) that a generic .fit()/.transform()
        illustration is actually load-bearing for the scope-limit fix
        below: the abstracted version (no method-name example at all)
        measured 10/10 still producing the false positive it's meant to
        prevent, while the version with the illustration measured 0/10.
        The distinction that matters is "don't name a library," not
        "don't use any concrete method name as an example."
        """
        lowered = API_SEQUENCE_SYSTEM_PROMPT.lower()
        for name in ("scikit-learn", "sklearn", "pandas"):
            self.assertNotIn(name, lowered)

    def test_states_the_missing_variable_scope_limit(self):
        """
        Regression guard for a real bug, confirmed 15/15 reproducible via
        a live user report before being root-caused: when a cell's own
        argument is a plain variable from a not-yet-executed cell (e.g.
        cell 7 has run, cell 8 - a plain assignment - has not, and cell 9
        calls a method using that variable), the model flagged cell 9
        itself, reasoning about the variable not being defined yet - a
        real fact, but the wrong bug class (a missing name, not a call-
        order violation, and out of this checker's scope entirely).
        Confirmed fixed (0/10, both in this exact state and in the
        correctly-flagged-elsewhere state) by this paragraph - and
        confirmed the general statement alone isn't enough, only the
        version with a worked counter-example fixed it (see
        test_does_not_name_specific_libraries's docstring).
        """
        self.assertIn("CRITICAL SCOPE LIMIT", API_SEQUENCE_SYSTEM_PROMPT)
        self.assertIn("plain variable", API_SEQUENCE_SYSTEM_PROMPT)

    def test_explains_the_not_yet_executed_marker(self):
        """
        Regression guard for a real gap found via direct user question
        ("doesn't cell 11 need fit to be run first - I can execute it
        directly"): a prerequisite call that only exists in the file but
        was never actually run must not count as satisfying anything.
        """
        self.assertIn("not yet executed", API_SEQUENCE_SYSTEM_PROMPT)
        self.assertIn("actually run", API_SEQUENCE_SYSTEM_PROMPT)

    def test_states_the_cell_id_attribution_rule(self):
        """
        Regression guard for a real, twice-reproduced misattribution bug:
        given a 3-hop chain (A creates an object, B is its required setup
        call and hasn't run, C calls something on it needing B), the
        model attached the finding to B (sometimes with a nonsensical
        second finding on the cell before that) instead of C - the cell
        that actually fails. Confirmed fixed (3/3) by this paragraph.
        """
        self.assertIn("cell_id must be the cell whose OWN code contains", API_SEQUENCE_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
