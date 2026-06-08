"""Unit tests for verifier module"""

import unittest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.verifier import SemanticScholarVerifier, VerificationResult


class TestSemanticScholarVerifier(unittest.TestCase):
    
    def setUp(self):
        self.verifier = SemanticScholarVerifier(cache_duration=60)
    
    def test_extract_query(self):
        """Test query extraction from claim text"""
        claim = "Our model outperforms state-of-the-art methods on ImageNet"
        query = self.verifier._extract_query(claim)
        
        # Should extract meaningful keywords
        self.assertIsInstance(query, str)
        self.assertGreater(len(query), 5)
        self.assertIn("outperforms", query.lower())
    
    def test_extract_query_short_text(self):
        """Test query extraction with very short text"""
        claim = "Test"
        query = self.verifier._extract_query(claim)
        # Should return empty or minimal query
        self.assertIsInstance(query, str)
    
    @patch('src.models.verifier.requests.get')
    def test_search_papers_success(self, mock_get):
        """Test successful paper search"""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"title": "Paper 1", "authors": [], "year": 2023},
                {"title": "Paper 2", "authors": [], "year": 2022}
            ]
        }
        mock_get.return_value = mock_response
        
        papers = self.verifier._search_papers("test query", limit=5)
        
        self.assertEqual(len(papers), 2)
        self.assertEqual(papers[0]["title"], "Paper 1")
    
    @patch('src.models.verifier.requests.get')
    def test_search_papers_failure(self, mock_get):
        """Test failed paper search"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        papers = self.verifier._search_papers("test query")
        
        self.assertEqual(papers, [])
    
    @patch('src.models.verifier.SemanticScholarVerifier._search_papers')
    def test_verify_with_results(self, mock_search):
        """Test verification when papers are found"""
        mock_search.return_value = [
            {"title": "Paper 1", "abstract": "Our method outperforms existing approaches"},
            {"title": "Paper 2", "abstract": "This paper presents a novel architecture"}
        ]
        
        result = self.verifier.verify("Our model outperforms baselines", {})
        
        self.assertIsInstance(result, VerificationResult)
        self.assertIsNotNone(result.explanation)
    
    def test_cache_functionality(self):
        """Test that caching works"""
        # First call should compute
        result1 = self.verifier.verify("Test claim", {})
        
        # Second call should use cache
        result2 = self.verifier.verify("Test claim", {})
        
        # Should be the same object (cached)
        self.assertEqual(result1.explanation, result2.explanation)
    
    def test_verification_result_structure(self):
        """Test VerificationResult dataclass"""
        result = VerificationResult(
            verified=True,
            confidence=0.85,
            supporting_sources=["Source 1", "Source 2"],
            contradicting_sources=[],
            explanation="Found supporting evidence",
            timestamp="2024-01-01T00:00:00"
        )
        
        self.assertTrue(result.verified)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(len(result.supporting_sources), 2)


if __name__ == "__main__":
    unittest.main()