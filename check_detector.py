# check_detector.py
from src.models.claim_detector import KeywordClaimDetector, BertClaimDetector

# Test keyword detector
keyword = KeywordClaimDetector()
print("Keyword detector test:")
claim_type, score = keyword.detect_claims("Our model outperforms baselines by 15%", {})
print(f"  Result: {claim_type} (score: {score})")

# Test BERT detector
bert = BertClaimDetector()
print("\nBERT detector test:")
if bert.is_available:
    claim_type, score = bert.detect_claims("Our model outperforms baselines by 15%", {})
    print(f"  Result: {claim_type} (score: {score})")
else:
    print("  BERT detector NOT available (model not trained)")