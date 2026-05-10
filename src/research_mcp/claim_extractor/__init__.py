"""Claim-extraction implementations of the `ClaimExtractor` protocol."""

from research_mcp.claim_extractor.fake import FakeClaimExtractor
from research_mcp.claim_extractor.spacy_extractor import SpacyClaimExtractor

__all__ = ["FakeClaimExtractor", "SpacyClaimExtractor"]
