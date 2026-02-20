from types import SimpleNamespace
from unittest import TestCase, mock
import uuid

from django.contrib.auth import get_user_model

from llm_chat.models import ChatThread
from llm_chat.services import ChatService
from llm_service.models import LLMCallLog


User = get_user_model()


def _llm_log():
    """Create a minimal valid LLMCallLog for tests (current schema has no parsed_json/succeeded)."""
    return LLMCallLog.objects.create(model="openai/gpt-5-nano")


class ChatServiceTest(TestCase):
    def setUp(self):
        # Ensure unique email per test run to avoid UNIQUE constraint issues
        email = f"user+{uuid.uuid4().hex}@example.com"
        self.user = User.objects.create_user(email=email, password="testpass")
        self.thread = ChatThread.objects.create(user=self.user, title="Service test")

    def test_stream_reply_raises_for_wrong_user(self):
        other = User.objects.create_user(email="other@example.com", password="testpass")
        service = ChatService()

        with self.assertRaises(PermissionError):
            list(
                service.stream_reply(
                    thread=self.thread,
                    user=other,
                    user_message="Hello",
                )
            )

    @mock.patch("llm_chat.services.LLMService")
    def test_stream_reply_yields_events_from_llm_service(self, MockLLMService):
        mock_service_instance = MockLLMService.return_value

        call_log = _llm_log()

        def fake_stream(**kwargs):
            yield ("response.output_text.delta", mock.Mock(delta="Hi"))
            yield ("final", {"call_log": call_log, "response": object()})

        mock_service_instance.call_llm_stream.side_effect = fake_stream

        service = ChatService()
        events = list(
            service.stream_reply(
                thread=self.thread,
                user=self.user,
                user_message="Hello",
            )
        )

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0][0], "response.output_text.delta")
        self.assertEqual(events[1][0], "final")

    @mock.patch("llm_chat.services.LLMService")
    def test_stream_reply_extracts_message_from_json_no_json_in_deltas(self, MockLLMService):
        """Test that deltas sent to frontend contain only message content, not JSON structure."""
        mock_service_instance = MockLLMService.return_value

        call_log = _llm_log()

        # Simulate JSON streaming: deltas arrive as JSON chunks
        def fake_json_stream(**kwargs):
            yield ("response.output_text.delta", mock.Mock(delta='{"message":'))
            yield ("response.output_text.delta", mock.Mock(delta='"Hi there!'))
            yield ("response.output_text.delta", mock.Mock(delta=' How can'))
            yield ("response.output_text.delta", mock.Mock(delta=' I help'))
            yield ("response.output_text.delta", mock.Mock(delta=' you today?"}'))
            yield ("final", {"call_log": call_log, "response": object()})

        mock_service_instance.call_llm_stream.side_effect = fake_json_stream

        service = ChatService()
        events = list(
            service.stream_reply(
                thread=self.thread,
                user=self.user,
                user_message="Hello",
            )
        )

        # Verify we got delta events and final
        delta_events = [e for e in events if e[0] == "response.output_text.delta"]
        self.assertGreater(len(delta_events), 0, "Should have delta events")
        
        # Verify NO delta contains JSON structure
        for event_type, event_obj in delta_events:
            delta = getattr(event_obj, "delta", "")
            # Delta should NOT contain JSON structure
            self.assertNotIn('{"message"', delta, f"Delta should not contain JSON: {repr(delta)}")
            self.assertNotIn('"message":', delta, f"Delta should not contain JSON key: {repr(delta)}")
            self.assertNotIn('}', delta, f"Delta should not contain closing brace: {repr(delta)}")
            # Delta should NOT start with JSON structure
            self.assertFalse(
                delta.strip().startswith(('{', '"message"', 'message":')),
                f"Delta should not start with JSON structure: {repr(delta)}"
            )
    
    @mock.patch("llm_chat.services.LLMService")
    def test_stream_reply_handles_complete_json_in_one_delta(self, MockLLMService):
        """Test that complete JSON in a single delta is properly extracted."""
        mock_service_instance = MockLLMService.return_value

        call_log = _llm_log()

        # Simulate complete JSON arriving in one delta
        def fake_stream(**kwargs):
            yield ("response.output_text.delta", mock.Mock(delta='{"message": "Hi there! How can I help you today?"}'))
            yield ("final", {"call_log": call_log, "response": object()})

        mock_service_instance.call_llm_stream.side_effect = fake_stream

        service = ChatService()
        events = list(
            service.stream_reply(
                thread=self.thread,
                user=self.user,
                user_message="Hello",
            )
        )

        # Verify we got a delta event
        delta_events = [e for e in events if e[0] == "response.output_text.delta"]
        self.assertEqual(len(delta_events), 1)
        
        # Verify the delta contains ONLY the message content, not JSON
        delta = getattr(delta_events[0][1], "delta", "")
        self.assertEqual(delta, "Hi there! How can I help you today?")
        self.assertNotIn('{"message"', delta)
        self.assertNotIn('"message":', delta)
        self.assertNotIn('}', delta)

    @mock.patch("llm_chat.services.LLMService")
    def test_generate_thread_title_success(self, MockLLMService):
        """Test that generate_thread_title creates a title for a new chat."""
        mock_service_instance = MockLLMService.return_value
        
        # Create a thread with default title
        thread = ChatThread.objects.create(user=self.user, title="New chat")
        
        # Mock successful LLM call (ChatService expects .succeeded, .parsed_json, .call_log)
        mock_service_instance.call_llm.return_value = SimpleNamespace(
            succeeded=True,
            parsed_json={"title": "Python Programming"},
            call_log=_llm_log(),
        )
        
        service = ChatService()
        title = service.generate_thread_title(
            thread=thread,
            user=self.user,
            user_message="What is Python?",
        )
        
        self.assertEqual(title, "Python Programming")
        thread.refresh_from_db()
        self.assertEqual(thread.title, "Python Programming")
        mock_service_instance.call_llm.assert_called_once()

    @mock.patch("llm_chat.services.LLMService")
    def test_generate_thread_title_skips_if_already_titled(self, MockLLMService):
        """Test that generate_thread_title doesn't regenerate if thread already has a custom title."""
        mock_service_instance = MockLLMService.return_value
        
        # Create a thread with custom title
        thread = ChatThread.objects.create(user=self.user, title="Custom Title")
        
        service = ChatService()
        title = service.generate_thread_title(
            thread=thread,
            user=self.user,
            user_message="What is Python?",
        )
        
        self.assertIsNone(title)
        thread.refresh_from_db()
        self.assertEqual(thread.title, "Custom Title")
        mock_service_instance.call_llm.assert_not_called()

    @mock.patch("llm_chat.services.LLMService")
    def test_generate_thread_title_handles_llm_failure(self, MockLLMService):
        """Test that generate_thread_title handles LLM failures gracefully."""
        mock_service_instance = MockLLMService.return_value
        
        # Create a thread with default title
        thread = ChatThread.objects.create(user=self.user, title="New chat")
        
        # Mock failed LLM call
        mock_service_instance.call_llm.return_value = SimpleNamespace(
            succeeded=False,
            parsed_json=None,
            call_log=_llm_log(),
        )
        
        service = ChatService()
        title = service.generate_thread_title(
            thread=thread,
            user=self.user,
            user_message="What is Python?",
        )
        
        self.assertIsNone(title)
        thread.refresh_from_db()
        self.assertEqual(thread.title, "New chat")  # Title unchanged

    @mock.patch("llm_chat.services.LLMService")
    def test_generate_thread_title_handles_exception(self, MockLLMService):
        """Test that generate_thread_title handles exceptions gracefully."""
        mock_service_instance = MockLLMService.return_value
        
        # Create a thread with default title
        thread = ChatThread.objects.create(user=self.user, title="New chat")
        
        # Mock exception
        mock_service_instance.call_llm.side_effect = Exception("API error")
        
        service = ChatService()
        title = service.generate_thread_title(
            thread=thread,
            user=self.user,
            user_message="What is Python?",
        )
        
        self.assertIsNone(title)
        thread.refresh_from_db()
        self.assertEqual(thread.title, "New chat")  # Title unchanged

    @mock.patch("llm_chat.services.LLMService")
    def test_generate_thread_title_success(self, MockLLMService):
        """Test that generate_thread_title creates a title for a new chat."""
        mock_service_instance = MockLLMService.return_value
        
        # Create a thread with default title
        thread = ChatThread.objects.create(user=self.user, title="New chat")
        
        # Mock successful LLM call (ChatService expects .succeeded, .parsed_json, .call_log)
        mock_service_instance.call_llm.return_value = SimpleNamespace(
            succeeded=True,
            parsed_json={"title": "Python Programming"},
            call_log=_llm_log(),
        )
        
        service = ChatService()
        title = service.generate_thread_title(
            thread=thread,
            user=self.user,
            user_message="What is Python?",
        )
        
        self.assertEqual(title, "Python Programming")
        thread.refresh_from_db()
        self.assertEqual(thread.title, "Python Programming")
        mock_service_instance.call_llm.assert_called_once()

    @mock.patch("llm_chat.services.LLMService")
    def test_generate_thread_title_skips_if_already_titled(self, MockLLMService):
        """Test that generate_thread_title doesn't regenerate if thread already has a custom title."""
        mock_service_instance = MockLLMService.return_value
        
        # Create a thread with custom title
        thread = ChatThread.objects.create(user=self.user, title="Custom Title")
        
        service = ChatService()
        title = service.generate_thread_title(
            thread=thread,
            user=self.user,
            user_message="What is Python?",
        )
        
        self.assertIsNone(title)
        thread.refresh_from_db()
        self.assertEqual(thread.title, "Custom Title")
        mock_service_instance.call_llm.assert_not_called()

    @mock.patch("llm_chat.services.LLMService")
    def test_generate_thread_title_handles_llm_failure(self, MockLLMService):
        """Test that generate_thread_title handles LLM failures gracefully."""
        mock_service_instance = MockLLMService.return_value
        
        # Create a thread with default title
        thread = ChatThread.objects.create(user=self.user, title="New chat")
        
        # Mock failed LLM call
        mock_service_instance.call_llm.return_value = SimpleNamespace(
            succeeded=False,
            parsed_json=None,
            call_log=_llm_log(),
        )
        
        service = ChatService()
        title = service.generate_thread_title(
            thread=thread,
            user=self.user,
            user_message="What is Python?",
        )
        
        self.assertIsNone(title)
        thread.refresh_from_db()
        self.assertEqual(thread.title, "New chat")  # Title unchanged

    @mock.patch("llm_chat.services.LLMService")
    def test_generate_thread_title_handles_exception(self, MockLLMService):
        """Test that generate_thread_title handles exceptions gracefully."""
        mock_service_instance = MockLLMService.return_value
        
        # Create a thread with default title
        thread = ChatThread.objects.create(user=self.user, title="New chat")
        
        # Mock exception
        mock_service_instance.call_llm.side_effect = Exception("API error")
        
        service = ChatService()
        title = service.generate_thread_title(
            thread=thread,
            user=self.user,
            user_message="What is Python?",
        )
        
        self.assertIsNone(title)
        thread.refresh_from_db()
        self.assertEqual(thread.title, "New chat")  # Title unchanged

