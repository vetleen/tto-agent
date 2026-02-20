"""
LiteLLM implementation of BaseLLMClient.
"""
import logging
from typing import Any

from llm_service.base import BaseLLMClient
from llm_service.conf import get_request_timeout

logger = logging.getLogger(__name__)


class LiteLLMClient(BaseLLMClient):
    """Client that delegates to litellm.completion / litellm.acompletion."""

    def completion(self, **kwargs: Any) -> Any:
        timeout = get_request_timeout()
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout
        import litellm
        return litellm.completion(**kwargs)

    def acompletion(self, **kwargs: Any) -> Any:
        timeout = get_request_timeout()
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout
        import litellm
        return litellm.acompletion(**kwargs)
