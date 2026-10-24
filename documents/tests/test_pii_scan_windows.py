"""The PII windowing shared by the production scan and the PII shadow
(documents.services.pii_scan.iter_pii_windows / pii_window_text)."""

from django.test import SimpleTestCase, TestCase, override_settings

from accounts.models import User
from documents.models import DataRoom, DataRoomDocument
from documents.services.pii_scan import iter_pii_windows, pii_window_text
from documents.tests._helpers import make_version


class WindowTextTests(SimpleTestCase):
    def test_heading_prefixes_the_chunk_and_chunks_are_blank_line_separated(self):
        text = pii_window_text([
            {"heading": " Parties ", "text": "Alice and Bob"},
            {"heading": "", "text": "no heading"},
            {"heading": None, "text": "none heading"},
        ])
        self.assertEqual(text, "Parties\nAlice and Bob\n\nno heading\n\nnone heading")

    def test_empty_window(self):
        self.assertEqual(pii_window_text([]), "")


class IterWindowsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="win@example.com", password="test1234")
        room = DataRoom.objects.create(name="Win", slug="win", created_by=self.user)
        self.doc = DataRoomDocument.objects.create(
            data_room=room, uploaded_by=self.user, original_filename="w.txt", status="ready",
        )

    def _version(self, tokens):
        return make_version(
            self.doc, chunks=[{"text": f"chunk {i}", "token_count": t, "heading": f"H{i}"} for i, t in enumerate(tokens)],
        )

    def test_windows_fill_to_the_budget_with_a_trailing_partial(self):
        version = self._version([10, 10, 10, 10, 5])
        windows = list(iter_pii_windows(version.id, budget=15))
        self.assertEqual([i for i, _ in windows], [0, 1, 2])
        self.assertEqual([[c["chunk_index"] for c in w] for _, w in windows], [[0, 1], [2, 3], [4]])
        self.assertEqual(windows[0][1][0]["heading"], "H0")
        self.assertEqual(windows[0][1][0]["text"], "chunk 0")

    def test_exact_budget_boundary_closes_the_window(self):
        version = self._version([10, 10])
        windows = list(iter_pii_windows(version.id, budget=10))
        self.assertEqual([[c["chunk_index"] for c in w] for _, w in windows], [[0], [1]])

    def test_everything_under_budget_is_one_window(self):
        version = self._version([3, 4])
        windows = list(iter_pii_windows(version.id, budget=6000))
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(len(windows[0][1]), 2)

    @override_settings(PII_SCAN_WINDOW_TOKENS=10)
    def test_default_budget_comes_from_settings(self):
        version = self._version([10, 10, 10])
        self.assertEqual(len(list(iter_pii_windows(version.id))), 3)

    def test_empty_version_yields_nothing(self):
        version = make_version(self.doc)
        self.assertEqual(list(iter_pii_windows(version.id)), [])

    def test_missing_token_count_counts_as_zero(self):
        version = make_version(self.doc, chunks=[{"text": "a", "token_count": 0}, {"text": "b", "token_count": 0}])
        windows = list(iter_pii_windows(version.id, budget=1))
        self.assertEqual(len(windows), 1)  # never reaches the budget → one trailing window
        self.assertEqual(len(windows[0][1]), 2)
