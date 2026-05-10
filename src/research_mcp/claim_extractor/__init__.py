"""Claim-extraction implementations of the `ClaimExtractor` protocol."""

from research_mcp.claim_extractor.anthropic_extractor import (
    AnthropicLLMClaimExtractor,
)
from research_mcp.claim_extractor.fake import FakeClaimExtractor
from research_mcp.claim_extractor.openai_extractor import OpenAILLMClaimExtractor
from research_mcp.claim_extractor.spacy_extractor import SpacyClaimExtractor

__all__ = [
    "AnthropicLLMClaimExtractor",
    "FakeClaimExtractor",
    "OpenAILLMClaimExtractor",
    "SpacyClaimExtractor",
]
