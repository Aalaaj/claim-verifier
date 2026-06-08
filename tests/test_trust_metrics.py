"""Unit tests for trust metrics calculator"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.trust_metrics import TrustMetricsCalculator


class TestTrustMetricsCalculator(unittest.TestCase):
    
    def setUp(self):
        self.calculator = TrustMetricsCalculator(
            theta=5.0,
            lambda_param=1.0,
            gamma_param=0.5
        )
    
    def test_internal_consistency_high_score(self):
        """Test internal consistency with high claim score"""
        result = self.calculator.calculate_internal_consistency(claim_score=0.9)
        self.assertGreater(result, 0.95)
        self.assertLessEqual(result, 1.0)
    
    def test_internal_consistency_low_score(self):
        """Test internal consistency with low claim score"""
        result = self.calculator.calculate_internal_consistency(claim_score=0.1)
        self.assertLess(result, 0.5)
        self.assertGreaterEqual(result, 0.0)
    
    def test_internal_consistency_mid_score(self):
        """Test internal consistency with medium claim score"""
        result = self.calculator.calculate_internal_consistency(claim_score=0.5)
        # Sigmoid of 2.5 ≈ 0.924
        self.assertAlmostEqual(result, 0.924, places=2)
    
    def test_external_verification_no_citations(self):
        """Test external verification with zero citations"""
        result = self.calculator.calculate_external_verification(
            num_citations=0, recent_citations=0
        )
        self.assertEqual(result, 0.0)
    
    def test_external_verification_many_citations(self):
        """Test external verification with many citations"""
        result = self.calculator.calculate_external_verification(
            num_citations=10, recent_citations=8
        )
        self.assertGreater(result, 0.8)
        self.assertLessEqual(result, 1.0)
    
    def test_external_verification_all_recent(self):
        """Test external verification with all citations recent"""
        result = self.calculator.calculate_external_verification(
            num_citations=5, recent_citations=5
        )
        # r = 1, so recency_component = 1 - γ*(0) = 1
        # quantity_component = 1 - e^(-5) ≈ 0.993
        self.assertAlmostEqual(result, 0.993, places=2)
    
    def test_citation_quality_no_citations(self):
        """Test citation quality with no citations"""
        result = self.calculator.calculate_citation_quality([], "text with no citations")
        self.assertEqual(result, 0.3)  # Default fallback
    
    def test_methodological_quality(self):
        """Test methodological quality detection"""
        text = "We conducted an experiment using cross-validation on a large dataset"
        result = self.calculator.calculate_methodological_quality(text)
        self.assertGreater(result, 0.3)
        self.assertLessEqual(result, 1.0)
    
    def test_methodological_quality_no_keywords(self):
        """Test methodological quality with no keywords"""
        text = "The results are shown in Figure 1."
        result = self.calculator.calculate_methodological_quality(text)
        self.assertEqual(result, 0.0)
    
    def test_reproducibility(self):
        """Test reproducibility detection"""
        text = "Our code and data are available on GitHub"
        result = self.calculator.calculate_reproducibility(text)
        self.assertGreater(result, 0.4)
    
    def test_structural_strength_perfect_match(self):
        """Test structural strength with perfect section match"""
        result = self.calculator.calculate_structural_strength(
            claim_type="Performance Comparison",
            section="Results",
            text="Our model outperforms baselines"
        )
        self.assertGreaterEqual(result, 0.8)
    
    def test_structural_strength_poor_match(self):
        """Test structural strength with poor section match"""
        result = self.calculator.calculate_structural_strength(
            claim_type="Methodology Claim",
            section="Conclusion",
            text="Future work will explore this direction"
        )
        self.assertLess(result, 0.6)
    
    def test_overall_trust(self):
        """Test overall trust score calculation"""
        metrics = {
            'internal_consistency': 0.9,
            'external_verification': 0.8,
            'citation_quality': 0.7,
            'methodological_soundness': 0.6,
            'reproducibility': 0.5
        }
        weights = {
            'internal_consistency': 0.15,
            'external_verification': 0.35,
            'citation_quality': 0.20,
            'methodological_soundness': 0.20,
            'reproducibility': 0.10
        }
        
        result = self.calculator.calculate_overall_trust(metrics, weights)
        
        expected = (0.9*0.15 + 0.8*0.35 + 0.7*0.20 + 0.6*0.20 + 0.5*0.10)
        self.assertAlmostEqual(result, expected, places=3)
        self.assertBetween(result, 0, 1)
    
    def assertBetween(self, value, low, high):
        """Assert value is between low and high inclusive"""
        self.assertGreaterEqual(value, low)
        self.assertLessEqual(value, high)


if __name__ == "__main__":
    unittest.main()