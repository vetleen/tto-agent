"""
Tests for markdown rendering in chat messages.

These tests verify that markdown content is correctly extracted from JSON responses
and stored in the database, ensuring the frontend can render it properly.
"""
import uuid
from unittest import TestCase, mock

from django.contrib.auth import get_user_model

from llm_chat.models import ChatMessage, ChatThread
from llm_chat.services import ChatService
from llm_service.models import LLMCallLog


User = get_user_model()


class MarkdownRenderingTest(TestCase):
    """Test markdown extraction and storage from LLM responses."""

    def setUp(self):
        email = f"user+{uuid.uuid4().hex}@example.com"
        self.user = User.objects.create_user(email=email, password="testpass")
        self.thread = ChatThread.objects.create(user=self.user, title="Markdown test")

    @mock.patch("llm_chat.services.LLMService")
    def test_unordered_list_markdown(self, MockLLMService):
        """Test that unordered list markdown is correctly extracted and stored."""
        mock_service_instance = MockLLMService.return_value

        markdown_content = "- Python\n- JavaScript\n- Rust"
        json_response = f'{{"message": "{markdown_content}"}}'

        call_log = LLMCallLog.objects.create(model="openai/gpt-5-nano")

        def fake_stream(**kwargs):
            # Simulate JSON streaming
            yield ("response.output_text.delta", mock.Mock(delta=json_response))
            yield ("final", {"call_log": call_log, "response": object()})

        mock_service_instance.call_llm_stream.side_effect = fake_stream

        service = ChatService()
        list(service.stream_reply(
            thread=self.thread,
            user=self.user,
            user_message="List programming languages",
        ))

        # Verify message was stored with correct markdown content
        assistant_msg = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT
        ).latest('created_at')

        self.assertEqual(assistant_msg.content, markdown_content)
        self.assertIn("- Python", assistant_msg.content)
        self.assertIn("- JavaScript", assistant_msg.content)
        self.assertIn("- Rust", assistant_msg.content)
        self.assertNotIn('{"message"', assistant_msg.content)
        self.assertNotIn('"message":', assistant_msg.content)

    @mock.patch("llm_chat.services.LLMService")
    def test_code_block_markdown(self, MockLLMService):
        """Test that code block markdown with syntax highlighting is correctly extracted."""
        mock_service_instance = MockLLMService.return_value

        markdown_content = 'Here\'s a simple "Hello, World!" script in Python:\n\n```python\nprint("Hello, world!")\n```\n\nTo run it:\n1. Save it as `hello.py`.\n2. In a terminal, run: `python hello.py`'
        # Escape for JSON
        json_content = markdown_content.replace('\n', '\\n').replace('"', '\\"')
        json_response = f'{{"message": "{json_content}"}}'

        call_log = LLMCallLog.objects.create(model="openai/gpt-5-nano")

        def fake_stream(**kwargs):
            yield ("response.output_text.delta", mock.Mock(delta=json_response))
            yield ("final", {"call_log": call_log, "response": object()})

        mock_service_instance.call_llm_stream.side_effect = fake_stream

        service = ChatService()
        list(service.stream_reply(
            thread=self.thread,
            user=self.user,
            user_message="Show me Python code",
        ))

        assistant_msg = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT
        ).latest('created_at')

        self.assertEqual(assistant_msg.content, markdown_content)
        self.assertIn("```python", assistant_msg.content)
        self.assertIn("print(", assistant_msg.content)
        self.assertIn("`hello.py`", assistant_msg.content)
        self.assertIn("1. Save it", assistant_msg.content)
        self.assertNotIn('{"message"', assistant_msg.content)

    @mock.patch("llm_chat.services.LLMService")
    def test_numbered_list_markdown(self, MockLLMService):
        """Test that numbered list markdown is correctly extracted."""
        mock_service_instance = MockLLMService.return_value

        markdown_content = "To get started:\n1. Install Python\n2. Create a virtual environment\n3. Install dependencies"
        json_content = markdown_content.replace('\n', '\\n')
        json_response = f'{{"message": "{json_content}"}}'

        call_log = LLMCallLog.objects.create(model="openai/gpt-5-nano")

        def fake_stream(**kwargs):
            yield ("response.output_text.delta", mock.Mock(delta=json_response))
            yield ("final", {"call_log": call_log, "response": object()})

        mock_service_instance.call_llm_stream.side_effect = fake_stream

        service = ChatService()
        list(service.stream_reply(
            thread=self.thread,
            user=self.user,
            user_message="How do I start?",
        ))

        assistant_msg = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT
        ).latest('created_at')

        self.assertEqual(assistant_msg.content, markdown_content)
        self.assertIn("1. Install Python", assistant_msg.content)
        self.assertIn("2. Create a virtual environment", assistant_msg.content)
        self.assertIn("3. Install dependencies", assistant_msg.content)

    @mock.patch("llm_chat.services.LLMService")
    def test_bold_and_italic_markdown(self, MockLLMService):
        """Test that bold and italic markdown is correctly extracted."""
        mock_service_instance = MockLLMService.return_value

        markdown_content = "This is **bold** text and this is *italic* text. You can also use ***bold italic***."
        json_response = f'{{"message": "{markdown_content}"}}'

        call_log = LLMCallLog.objects.create(model="openai/gpt-5-nano")

        def fake_stream(**kwargs):
            yield ("response.output_text.delta", mock.Mock(delta=json_response))
            yield ("final", {"call_log": call_log, "response": object()})

        mock_service_instance.call_llm_stream.side_effect = fake_stream

        service = ChatService()
        list(service.stream_reply(
            thread=self.thread,
            user=self.user,
            user_message="Show formatting",
        ))

        assistant_msg = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT
        ).latest('created_at')

        self.assertEqual(assistant_msg.content, markdown_content)
        self.assertIn("**bold**", assistant_msg.content)
        self.assertIn("*italic*", assistant_msg.content)
        self.assertIn("***bold italic***", assistant_msg.content)

    @mock.patch("llm_chat.services.LLMService")
    def test_headers_markdown(self, MockLLMService):
        """Test that header markdown is correctly extracted."""
        mock_service_instance = MockLLMService.return_value

        markdown_content = "# Main Title\n\n## Section 1\n\n### Subsection\n\nRegular text here."
        json_content = markdown_content.replace('\n', '\\n')
        json_response = f'{{"message": "{json_content}"}}'

        call_log = LLMCallLog.objects.create(model="openai/gpt-5-nano")

        def fake_stream(**kwargs):
            yield ("response.output_text.delta", mock.Mock(delta=json_response))
            yield ("final", {"call_log": call_log, "response": object()})

        mock_service_instance.call_llm_stream.side_effect = fake_stream

        service = ChatService()
        list(service.stream_reply(
            thread=self.thread,
            user=self.user,
            user_message="Create headers",
        ))

        assistant_msg = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT
        ).latest('created_at')

        self.assertEqual(assistant_msg.content, markdown_content)
        self.assertIn("# Main Title", assistant_msg.content)
        self.assertIn("## Section 1", assistant_msg.content)
        self.assertIn("### Subsection", assistant_msg.content)

    @mock.patch("llm_chat.services.LLMService")
    def test_inline_code_markdown(self, MockLLMService):
        """Test that inline code markdown is correctly extracted."""
        mock_service_instance = MockLLMService.return_value

        markdown_content = "Use the `print()` function to output text. You can also use `console.log()` in JavaScript."
        json_response = f'{{"message": "{markdown_content}"}}'

        call_log = LLMCallLog.objects.create(model="openai/gpt-5-nano")

        def fake_stream(**kwargs):
            yield ("response.output_text.delta", mock.Mock(delta=json_response))
            yield ("final", {"call_log": call_log, "response": object()})

        mock_service_instance.call_llm_stream.side_effect = fake_stream

        service = ChatService()
        list(service.stream_reply(
            thread=self.thread,
            user=self.user,
            user_message="Show inline code",
        ))

        assistant_msg = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT
        ).latest('created_at')

        self.assertEqual(assistant_msg.content, markdown_content)
        self.assertIn("`print()`", assistant_msg.content)
        self.assertIn("`console.log()`", assistant_msg.content)

    @mock.patch("llm_chat.services.LLMService")
    def test_links_markdown(self, MockLLMService):
        """Test that link markdown is correctly extracted."""
        mock_service_instance = MockLLMService.return_value

        markdown_content = "Check out [Django documentation](https://docs.djangoproject.com) for more info."
        json_response = f'{{"message": "{markdown_content}"}}'

        call_log = LLMCallLog.objects.create(model="openai/gpt-5-nano")

        def fake_stream(**kwargs):
            yield ("response.output_text.delta", mock.Mock(delta=json_response))
            yield ("final", {"call_log": call_log, "response": object()})

        mock_service_instance.call_llm_stream.side_effect = fake_stream

        service = ChatService()
        list(service.stream_reply(
            thread=self.thread,
            user=self.user,
            user_message="Add a link",
        ))

        assistant_msg = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT
        ).latest('created_at')

        self.assertEqual(assistant_msg.content, markdown_content)
        self.assertIn("[Django documentation]", assistant_msg.content)
        self.assertIn("(https://docs.djangoproject.com)", assistant_msg.content)

    @mock.patch("llm_chat.services.LLMService")
    def test_blockquote_markdown(self, MockLLMService):
        """Test that blockquote markdown is correctly extracted."""
        mock_service_instance = MockLLMService.return_value

        markdown_content = "Here's a quote:\n\n> The only way to do great work is to love what you do.\n\nThat's inspiring!"
        json_content = markdown_content.replace('\n', '\\n')
        json_response = f'{{"message": "{json_content}"}}'

        call_log = LLMCallLog.objects.create(model="openai/gpt-5-nano")

        def fake_stream(**kwargs):
            yield ("response.output_text.delta", mock.Mock(delta=json_response))
            yield ("final", {"call_log": call_log, "response": object()})

        mock_service_instance.call_llm_stream.side_effect = fake_stream

        service = ChatService()
        list(service.stream_reply(
            thread=self.thread,
            user=self.user,
            user_message="Add a quote",
        ))

        assistant_msg = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT
        ).latest('created_at')

        self.assertEqual(assistant_msg.content, markdown_content)
        self.assertIn("> The only way", assistant_msg.content)

    @mock.patch("llm_chat.services.LLMService")
    def test_complex_mixed_markdown(self, MockLLMService):
        """Test complex markdown with multiple elements mixed together."""
        mock_service_instance = MockLLMService.return_value

        markdown_content = """# Python Guide

## Getting Started

Here's a **simple** example:

```python
def hello():
    print("Hello, world!")
```

### Steps

1. Install Python from [python.org](https://python.org)
2. Use `python --version` to verify
3. Create a file with `.py` extension

> Remember: Practice makes perfect!

*Note*: This is just the beginning."""
        json_content = markdown_content.replace('\n', '\\n').replace('"', '\\"')
        json_response = f'{{"message": "{json_content}"}}'

        call_log = LLMCallLog.objects.create(model="openai/gpt-5-nano")

        def fake_stream(**kwargs):
            yield ("response.output_text.delta", mock.Mock(delta=json_response))
            yield ("final", {"call_log": call_log, "response": object()})

        mock_service_instance.call_llm_stream.side_effect = fake_stream

        service = ChatService()
        list(service.stream_reply(
            thread=self.thread,
            user=self.user,
            user_message="Create a guide",
        ))

        assistant_msg = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT
        ).latest('created_at')

        self.assertEqual(assistant_msg.content, markdown_content)
        # Verify all markdown elements are present
        self.assertIn("# Python Guide", assistant_msg.content)
        self.assertIn("## Getting Started", assistant_msg.content)
        self.assertIn("**simple**", assistant_msg.content)
        self.assertIn("```python", assistant_msg.content)
        self.assertIn("1. Install Python", assistant_msg.content)
        self.assertIn("[python.org]", assistant_msg.content)
        self.assertIn("`python --version`", assistant_msg.content)
        self.assertIn("> Remember:", assistant_msg.content)
        self.assertIn("*Note*:", assistant_msg.content)
        # Verify JSON structure is NOT present
        self.assertNotIn('{"message"', assistant_msg.content)
        self.assertNotIn('"message":', assistant_msg.content)

    @mock.patch("llm_chat.services.LLMService")
    def test_markdown_with_escaped_characters(self, MockLLMService):
        """Test markdown extraction handles escaped characters correctly."""
        mock_service_instance = MockLLMService.return_value

        # Content with quotes and newlines that need escaping in JSON
        markdown_content = 'He said "Hello!"\n\nAnd then:\n- Item 1\n- Item 2'
        json_content = markdown_content.replace('\n', '\\n').replace('"', '\\"')
        json_response = f'{{"message": "{json_content}"}}'

        call_log = LLMCallLog.objects.create(model="openai/gpt-5-nano")

        def fake_stream(**kwargs):
            yield ("response.output_text.delta", mock.Mock(delta=json_response))
            yield ("final", {"call_log": call_log, "response": object()})

        mock_service_instance.call_llm_stream.side_effect = fake_stream

        service = ChatService()
        list(service.stream_reply(
            thread=self.thread,
            user=self.user,
            user_message="Test escaping",
        ))

        assistant_msg = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT
        ).latest('created_at')

        self.assertEqual(assistant_msg.content, markdown_content)
        self.assertIn('"Hello!"', assistant_msg.content)
        self.assertIn("- Item 1", assistant_msg.content)
        self.assertIn("- Item 2", assistant_msg.content)

    @mock.patch("llm_chat.services.LLMService")
    def test_streaming_json_extraction_preserves_markdown(self, MockLLMService):
        """Test that markdown is correctly extracted when JSON arrives in chunks."""
        mock_service_instance = MockLLMService.return_value

        markdown_content = "- Python\n- JavaScript\n- Rust"
        json_parts = [
            '{"message":',
            '"',
            '- Python',
            '\\n',
            '- JavaScript',
            '\\n',
            '- Rust',
            '"}'
        ]

        call_log = LLMCallLog.objects.create(model="openai/gpt-5-nano")

        def fake_stream(**kwargs):
            for part in json_parts:
                yield ("response.output_text.delta", mock.Mock(delta=part))
            yield ("final", {"call_log": call_log, "response": object()})

        mock_service_instance.call_llm_stream.side_effect = fake_stream

        service = ChatService()
        events = list(service.stream_reply(
            thread=self.thread,
            user=self.user,
            user_message="List languages",
        ))

        # Verify deltas don't contain JSON structure
        delta_events = [e for e in events if e[0] == "response.output_text.delta"]
        for event_type, event_obj in delta_events:
            delta = getattr(event_obj, "delta", "")
            self.assertNotIn('{"message"', delta)
            self.assertNotIn('"message":', delta)
            self.assertNotIn('}', delta)

        # Verify final message content is correct
        assistant_msg = ChatMessage.objects.filter(
            thread=self.thread,
            role=ChatMessage.Role.ASSISTANT
        ).latest('created_at')

        self.assertEqual(assistant_msg.content, markdown_content)
        self.assertIn("- Python", assistant_msg.content)
        self.assertIn("- JavaScript", assistant_msg.content)
        self.assertIn("- Rust", assistant_msg.content)
