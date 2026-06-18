"""Tests for chat.repetition_guard.is_degenerate."""

from django.test import SimpleTestCase

from chat.repetition_guard import is_degenerate

# A long, varied paragraph (> 1000 chars) with no repeated sentence runs.
_NORMAL_PROSE = (
    "Technology transfer offices evaluate new inventions for commercial potential. "
    "An invention disclosure starts the process, capturing what was made and by whom. "
    "Staff then run a prior-art search to gauge novelty against existing patents. "
    "A market assessment estimates demand, competitors, and the likely licensees. "
    "If the case is strong, the office files a provisional patent application. "
    "Researchers keep developing the work while the legal timeline advances. "
    "Licensing managers reach out to companies that could bring the product to market. "
    "Negotiations cover royalties, milestones, field-of-use terms, and equity in spinouts. "
    "Some inventions become startups, and the office may take a small founding stake. "
    "Throughout, careful records keep the university compliant with its funding rules. "
    "Success is measured in licenses signed, revenue returned, and products launched. "
    "Each case differs, so judgement and domain knowledge matter as much as procedure. "
    "Early conversations with the inventor shape how aggressively to pursue protection. "
    "External counsel drafts claims while the office tracks deadlines and renewal fees. "
    "Industry scouts and alumni networks often surface the most promising licensees. "
    "When a deal closes, revenue is shared with inventors, departments, and the fund. "
    "Periodic portfolio reviews prune dormant cases so resources follow live opportunities. "
)

# A varied list of distinct items (> 1000 chars) — structure repeats, content does not.
_VARIED_LIST = "\n".join(
    f"{i}. Action item {i}: review the {topic} and report findings to the committee."
    for i, topic in enumerate(
        [
            "budget", "timeline", "patent landscape", "market sizing", "licensing terms",
            "regulatory path", "competitor analysis", "founding team", "cap table",
            "grant compliance", "milestone schedule", "royalty model", "exit options",
            "IP assignment", "publication plan", "prototype status", "supply chain",
        ],
        start=1,
    )
)


class IsDegenerateTests(SimpleTestCase):
    def test_short_text_is_never_degenerate(self):
        # Repetitive, but under the 1000-char floor → not judged.
        self.assertFalse(is_degenerate("Lets do this. Lets go. " * 5))

    def test_repetition_loop_is_degenerate(self):
        loop = "Lets do this. Lets go. Let me know what you think. " * 80
        self.assertTrue(len(loop) > 1000)
        self.assertTrue(is_degenerate(loop))

    def test_normal_prose_is_not_degenerate(self):
        self.assertTrue(len(_NORMAL_PROSE) > 1000)
        self.assertFalse(is_degenerate(_NORMAL_PROSE))

    def test_varied_list_is_not_degenerate(self):
        self.assertTrue(len(_VARIED_LIST) > 1000)
        self.assertFalse(is_degenerate(_VARIED_LIST))

    def test_empty_string_is_not_degenerate(self):
        self.assertFalse(is_degenerate(""))
