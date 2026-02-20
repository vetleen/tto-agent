"""Constants for the chat app (e.g. default model, key models for dropdown)."""

# Default model for chat when none is selected (must be in LLM_ALLOWED_MODELS).
CHAT_DEFAULT_MODEL = "moonshot/kimi-k2.5"

# Key models shown in the chat model dropdown: (model_id, display label).
# Only entries whose model_id is in LLM_ALLOWED_MODELS are shown.
CHAT_KEY_MODELS = [
    ("moonshot/kimi-k2.5", "Kimi K2.5"),
    ("moonshot/kimi-k2-thinking", "Kimi K2 Thinking"),
    ("openai/gpt-5.2", "GPT-5.2"),
    ("openai/gpt-5-mini", "GPT-5 Mini"),
    ("openai/gpt-5-nano", "GPT-5 Nano"),
    ("anthropic/claude-sonnet-4-5-20250929", "Claude Sonnet 4.5"),
    ("gemini/gemini-3-flash-preview", "Gemini 3 Flash"),
]
