"""AgentStage drop-in LLM clients.

These wrap the underlying SDK call (anthropic, google-genai, openai),
intercept streaming events, run the detector live, dispatch DataHints
to the stager, and forward everything to the caller unchanged.

All three take the same constructor arguments (`api_key`, `stager`,
`workspace_prior`, `ruleset`) and share the rule-matching and staging logic
in `agentstage.client.dispatch`. Import the one matching your provider:

    from agentstage.client.anthropic import AnthropicClient
    from agentstage.client.gemini import GeminiClient
    from agentstage.client.openai import OpenAIClient

`OpenAIClient` targets any OpenAI-compatible endpoint. Staging depends on the
server surfacing reasoning text; see that module's docstring.
"""

from .anthropic import AnthropicClient, ToolCall
from .dispatch import RuleDispatcher, tier_for_size

__all__ = ["AnthropicClient", "RuleDispatcher", "ToolCall", "tier_for_size"]
