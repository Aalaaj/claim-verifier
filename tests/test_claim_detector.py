"""Unit tests for claim detector"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.claim_detector import KeywordClaimDetector


class TestKeywordClaimDetector(unittest.TestCase):
    
    def setUp(self):
        self.detector = KeywordClaimDetector()
    
    def test_detects_performance_claim(self):
        text = "Our model outperforms the state-of-the-art by 15%"
        claim_type, score = self.detector.detect_claims(text, {})
        self.assertEqual(claim_type, "Performance Comparison")
        self.assertGreater(score, 0.5)
    
    def test_detects_novelty_claim(self):
        text = "We propose a novel architecture for transformer networks"
        claim_type, score = self.detector.detect_claims(text, {})
        self.assertEqual(claim_type, "Methodology Claim")
        self.assertGreater(score, 0.5)
    
    def test_rejects_non_claim(self):
        text = "This paper is organized as follows."
        claim_type, score = self.detector.detect_claims(text, {})
        self.assertEqual(claim_type, "No Claim")
        self.assertEqual(score, 0.0)
    
    def test_detects_statistical_claim(self):
        text = "The results showed a statistically significant difference (p < 0.05)"
        claim_type, score = self.detector.detect_claims(text, {})
        self.assertIn(claim_type, ["Statistical Claim", "General Finding"])
        self.assertGreater(score, 0.5)


if __name__ == "__main__":
    unittest.main()