import unittest

from nbfix.llm.stale_context_builder import StaleDetectionContext
from nbfix.llm.stale_prompts import STALE_SYSTEM_PROMPT, build_stale_user_prompt


class TestBuildStaleUserPrompt(unittest.TestCase):
    def test_includes_original_and_current_code_for_edited_cell(self):
        context = StaleDetectionContext(
            edited_cell=0, original_code="x = 10", current_code="x = 99",
            cells={0: "x = 99", 1: "y = x + 1"},
        )
        prompt = build_stale_user_prompt(context)
        self.assertIn("x = 10", prompt)
        self.assertIn("x = 99", prompt)
        self.assertIn("Cell 0", prompt)

    def test_includes_every_cells_current_code(self):
        context = StaleDetectionContext(
            edited_cell=0, original_code="x = 10", current_code="x = 99",
            cells={0: "x = 99", 1: "y = x + 1", 2: "z = y + 1"},
        )
        prompt = build_stale_user_prompt(context)
        self.assertIn("Cell 1", prompt)
        self.assertIn("y = x + 1", prompt)
        self.assertIn("Cell 2", prompt)
        self.assertIn("z = y + 1", prompt)


class TestStaleSystemPrompt(unittest.TestCase):
    def test_documents_stale_cells_schema(self):
        """
        Schema is a bare cell_id int list, not a {cell_id, message}
        object - the message field was dropped to cut completion tokens
        (measured ~7x) and latency (~3x) for live, interactive use. See
        this module's docstring for the accuracy tradeoff that came with
        it, and stale_result_mapping.py for the fixed replacement message
        surfaced in the editor instead.
        """
        self.assertIn("stale_cells", STALE_SYSTEM_PROMPT)
        self.assertIn("cell_id", STALE_SYSTEM_PROMPT)
        self.assertNotIn("message", STALE_SYSTEM_PROMPT)

    def test_asks_the_operational_question_not_the_passive_one(self):
        """
        Regression guard for the specific framing choice this module is
        built around (see its own docstring and experiments.md finding
        13) - the prompt must ask whether re-executing a cell now would
        be correct, not merely whether its displayed value looks old.
        The passive framing was measured to make both a small local
        model and a separate Claude session independently over-include
        the cell reading the edited variable directly.
        """
        self.assertIn("RIGHT NOW", STALE_SYSTEM_PROMPT)
        self.assertIn("is fine", STALE_SYSTEM_PROMPT)
        self.assertIn("to run next", STALE_SYSTEM_PROMPT)

    def test_states_the_hop_count_rule_mechanically(self):
        """
        Regression guard for a live failure: even with the operational
        framing above, a real qwen2.5-coder:14b response flagged the cell
        directly reading the edited variable as stale ("has not been
        updated"), the exact mistake the operational framing was meant to
        prevent. The real StaleCellAnalysis never reasons in prose - it
        propagates an integer hop count and only flags a cell at K=2+ (see
        code_impact_abs_state.py's condition()). The prompt must state
        that same rule mechanically, including the explicit "hop 1 is
        exempt" carve-out, rather than relying on free-form reasoning to
        rediscover it.
        """
        self.assertIn("Hop 1", STALE_SYSTEM_PROMPT)
        self.assertIn("never flag it", STALE_SYSTEM_PROMPT)
        self.assertIn("Hop 2", STALE_SYSTEM_PROMPT)

    def test_question_is_phrased_as_a_safety_call_not_a_temporal_one(self):
        """
        "Which cells cannot yet be correctly re-executed?" reads equally
        well as "which cells are behind" - which is exactly the wrong
        answer a live model gave. The question must instead ask which
        cells should NOT be executed, an explicit safety/correctness call.
        """
        context = StaleDetectionContext(
            edited_cell=0, original_code="x = 10", current_code="x = 99",
            cells={0: "x = 99", 1: "y = x + 1"},
        )
        prompt = build_stale_user_prompt(context)
        self.assertIn("should NOT be executed", prompt)


if __name__ == "__main__":
    unittest.main()
