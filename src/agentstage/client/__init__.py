"""AgentStage drop-in LLM clients.

These wrap the underlying SDK call (anthropic, openai, google-genai),
intercept streaming events, run the detector live, dispatch DataHints
to the stager, and forward everything to the caller unchanged.

Anthropic and Gemini wrappers are implemented; import GeminiClient
from `agentstage.client.gemini`.
"""

from .anthropic import AnthropicClient, ToolCall

__all__ = ["AnthropicClient", "ToolCall"]
