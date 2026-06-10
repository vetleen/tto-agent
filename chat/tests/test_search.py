"""Tests for the sidebar chat search API."""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from chat.models import ChatMessage, ChatThread

User = get_user_model()


class ChatSearchTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="searcher@example.com", password="testpass")
        self.user.email_verified = True
        self.user.save(update_fields=["email_verified"])
        self.client.force_login(self.user)
        self.url = reverse("chat_search")

    def _results(self, q):
        response = self.client.get(self.url, {"q": q})
        self.assertEqual(response.status_code, 200)
        return response.json()["results"]

    def test_login_required(self):
        self.client.logout()
        response = self.client.get(self.url, {"q": "anything"})
        self.assertEqual(response.status_code, 302)

    def test_empty_query_returns_no_results(self):
        ChatThread.objects.create(created_by=self.user, title="Licensing terms")
        self.assertEqual(self._results(""), [])
        self.assertEqual(self._results("   "), [])

    def test_title_match(self):
        thread = ChatThread.objects.create(created_by=self.user, title="Licensing terms NTNU")
        results = self._results("licensing")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], str(thread.id))
        self.assertEqual(results[0]["title"], "Licensing terms NTNU")
        self.assertIsNone(results[0]["snippet"])
        self.assertFalse(results[0]["is_archived"])

    def test_content_match_returns_snippet(self):
        thread = ChatThread.objects.create(created_by=self.user, title="Untitled")
        ChatMessage.objects.create(
            thread=thread, role=ChatMessage.Role.USER,
            content="What are the royalty rates for the spinout deal?",
        )
        results = self._results("royalty")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], str(thread.id))
        self.assertIn("royalty", results[0]["snippet"])

    def test_assistant_messages_searchable(self):
        thread = ChatThread.objects.create(created_by=self.user, title="Untitled")
        ChatMessage.objects.create(
            thread=thread, role=ChatMessage.Role.ASSISTANT,
            content="The patent expires in 2031.",
        )
        results = self._results("patent expires")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], str(thread.id))

    def test_tool_and_system_messages_not_searchable(self):
        thread = ChatThread.objects.create(created_by=self.user, title="Untitled")
        ChatMessage.objects.create(
            thread=thread, role=ChatMessage.Role.TOOL, content="zebra tool output blob",
        )
        ChatMessage.objects.create(
            thread=thread, role=ChatMessage.Role.SYSTEM, content="zebra system prompt",
        )
        self.assertEqual(self._results("zebra"), [])

    def test_hidden_messages_not_searchable(self):
        thread = ChatThread.objects.create(created_by=self.user, title="Untitled")
        ChatMessage.objects.create(
            thread=thread, role=ChatMessage.Role.USER,
            content="walrus seeding context", is_hidden_from_user=True,
        )
        self.assertEqual(self._results("walrus"), [])

    def test_other_users_threads_excluded(self):
        other = User.objects.create_user(email="other@example.com", password="testpass")
        other_thread = ChatThread.objects.create(created_by=other, title="Quantum licensing")
        ChatMessage.objects.create(
            thread=other_thread, role=ChatMessage.Role.USER, content="quantum secrets",
        )
        self.assertEqual(self._results("quantum"), [])

    def test_archived_threads_included_and_flagged(self):
        thread = ChatThread.objects.create(
            created_by=self.user, title="Old quantum chat", is_archived=True,
        )
        results = self._results("quantum")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], str(thread.id))
        self.assertTrue(results[0]["is_archived"])

    def test_title_matches_rank_above_content_matches(self):
        content_thread = ChatThread.objects.create(created_by=self.user, title="Untitled")
        ChatMessage.objects.create(
            thread=content_thread, role=ChatMessage.Role.USER,
            content="tell me about gravity waves",
        )
        # Created later, so more recently updated — but title match still wins.
        title_thread = ChatThread.objects.create(created_by=self.user, title="Gravity research")
        results = self._results("gravity")
        self.assertEqual([r["id"] for r in results], [str(title_thread.id), str(content_thread.id)])
        self.assertIsNone(results[0]["snippet"])
        self.assertIsNotNone(results[1]["snippet"])

    def test_thread_matching_title_and_content_appears_once(self):
        thread = ChatThread.objects.create(created_by=self.user, title="Gravity research")
        ChatMessage.objects.create(
            thread=thread, role=ChatMessage.Role.USER, content="gravity is fascinating",
        )
        results = self._results("gravity")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], str(thread.id))
        self.assertIsNone(results[0]["snippet"])

    def test_short_query_searches_titles_only(self):
        title_thread = ChatThread.objects.create(created_by=self.user, title="AI strategy")
        content_thread = ChatThread.objects.create(created_by=self.user, title="Untitled")
        ChatMessage.objects.create(
            thread=content_thread, role=ChatMessage.Role.USER, content="AI is everywhere",
        )
        results = self._results("AI")
        self.assertEqual([r["id"] for r in results], [str(title_thread.id)])

    def test_one_result_per_thread_with_multiple_matching_messages(self):
        thread = ChatThread.objects.create(created_by=self.user, title="Untitled")
        ChatMessage.objects.create(
            thread=thread, role=ChatMessage.Role.USER, content="first mention of falcon",
        )
        ChatMessage.objects.create(
            thread=thread, role=ChatMessage.Role.USER, content="second mention of falcon",
        )
        results = self._results("falcon")
        self.assertEqual(len(results), 1)
        # Snippet comes from the most recent matching message.
        self.assertIn("second", results[0]["snippet"])

    def test_search_is_case_insensitive(self):
        ChatThread.objects.create(created_by=self.user, title="Licensing Terms")
        self.assertEqual(len(self._results("LICENSING")), 1)

    def test_long_content_snippet_is_windowed(self):
        thread = ChatThread.objects.create(created_by=self.user, title="Untitled")
        content = ("padding " * 50) + "needle in the haystack" + (" trailing" * 50)
        ChatMessage.objects.create(thread=thread, role=ChatMessage.Role.USER, content=content)
        results = self._results("needle")
        snippet = results[0]["snippet"]
        self.assertIn("needle", snippet)
        self.assertLess(len(snippet), 200)
        self.assertTrue(snippet.startswith("…"))
        self.assertTrue(snippet.endswith("…"))
