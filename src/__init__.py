"""Thesis Claim Analyzer - A framework for detecting and analyzing claims in research papers"""

__version__ = "1.0.0"
__author__ = "Aalaa"

from src.pipeline.analyzer import ThesisClaimAnalyzer
from src.models.claim_detector import EnsembleDetector, KeywordClaimDetector, BertClaimDetector
from src.models.verifier import SemanticScholarVerifier

__all__ = [
    "ThesisClaimAnalyzer",
    "EnsembleDetector",
    "KeywordClaimDetector", 
    "BertClaimDetector",
    "SemanticScholarVerifier",
]