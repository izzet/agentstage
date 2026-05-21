"""AgentStage drop-in LLM clients.

These wrap the underlying SDK call (anthropic, openai, google-genai),
intercept streaming events, run the detector live, dispatch DataHints
to the stager, and forward everything to the caller unchanged.

For Path A smoke (Day 2), only AnthropicClient is implemented. The
Gemini and HTTP wrappers land on Days 3-4 (see TASKS.md).
"""

from .anthropic import AnthropicClient, ToolCall

__all__ = ["AnthropicClient", "ToolCall"]
