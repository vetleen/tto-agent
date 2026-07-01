"""Built-in tools — web search and fetch."""

from django.conf import settings

from llm.tools.registry import get_tool_registry
from llm.tools.web_fetch import WebFetchTool

registry = get_tool_registry()
registry.register_tool(WebFetchTool())

if getattr(settings, "BRAVE_SEARCH_API_KEY", None):
    from llm.tools.brave_search import BraveSearchTool
    from llm.tools.web_search_and_read import WebSearchAndReadTool
    registry.register_tool(BraveSearchTool())
    registry.register_tool(WebSearchAndReadTool())

if getattr(settings, "EPO_OPS_KEY", None) and getattr(settings, "EPO_OPS_SECRET", None):
    from llm.tools.epo_ops import (
        PatentEpoOpsFamilyTool,
        PatentEpoOpsGetTool,
        PatentEpoOpsSearchTool,
    )
    registry.register_tool(PatentEpoOpsSearchTool())
    registry.register_tool(PatentEpoOpsGetTool())
    registry.register_tool(PatentEpoOpsFamilyTool())

__all__: list[str] = []
