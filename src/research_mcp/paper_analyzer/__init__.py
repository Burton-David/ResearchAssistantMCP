"""PaperAnalyzer implementations."""

from research_mcp.paper_analyzer.anthropic_analyzer import (
    AnthropicLLMPaperAnalyzer,
)
from research_mcp.paper_analyzer.fake import FakePaperAnalyzer
from research_mcp.paper_analyzer.openai_analyzer import OpenAILLMPaperAnalyzer

__all__ = [
    "AnthropicLLMPaperAnalyzer",
    "FakePaperAnalyzer",
    "OpenAILLMPaperAnalyzer",
]
