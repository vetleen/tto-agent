"""Built-in tools — web search and fetch."""

from django.conf import settings

from llm.tools.registry import get_tool_registry
from llm.tools.web_fetch import WebFetchTool

registry = get_tool_registry()
registry.register_tool(WebFetchTool())

if getattr(settings, "BRAVE_SEARCH_API_KEY", None):
    from llm.tools.brave_search import BraveSearchTool
    registry.register_tool(BraveSearchTool())

__all__: list[str] = []
