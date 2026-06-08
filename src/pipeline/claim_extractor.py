"""Claim extraction from parsed documents"""

import re
from typing import List, Dict, Tuple, Optional
from collections import Counter

from src.models.claim_detector import ClaimDetector, KeywordClaimDetector
from src.models.verifier import SemanticScholarVerifier


class ClaimExtractor:
    """Extracts claims from parsed documents"""
    
    def __init__(self, detector: Optional[ClaimDetector] = None):
        self.detector = detector or KeywordClaimDetector()
        self.verifier = SemanticScholarVerifier()
        
        # Configuration
        self.min_claim_length = 40
        self.max_claims_per_paper = 500
        
    def extract_claims(self, parsed_doc, sections: Dict = None) -> List[Dict]:
        """
        Extract claims from a parsed document
        
        Args:
            parsed_doc: ParsedDocument object
            sections: Section boundaries (optional)
            
        Returns:
            List of claim dictionaries
        """
        claims = []
        seen_texts = set()
        
        # Use provided sections or empty dict
        sections = sections or {}
        
        # Process each page
        doc = fitz.open(parsed_doc.file_path)
        
        for page_num in range(len(doc)):
            section = self._get_section_for_page(page_num + 1, sections)
            page = doc.load_page(page_num)
            
            # Extract text blocks
            blocks = page.get_text("blocks")
            
            for block in blocks:
                text = block[4].strip()
                
                # Filter by length and duplicates
                if len(text) < self.min_claim_length or text in seen_texts:
                    continue
                
                # Clean text
                text = self._clean_text(text)
                
                # Detect claim
                claim_type, claim_score = self.detector.detect_claims(text, {'section': section})
                
                if claim_score >= 0.3:  # Minimum threshold
                    claim = {
                        'text': text,
                        'page': page_num + 1,
                        'section': section.capitalize() if section != "Unknown" else "Unknown",
                        'claim_type': claim_type,
                        'claim_score': round(claim_score, 2),
                        'verification_status': 'Pending',
                        'confidence_score': 0.0,
                        'explanation': ''
                    }
                    
                    claims.append(claim)
                    seen_texts.add(text)
                    
                    if len(claims) >= self.max_claims_per_paper:
                        break
            
            if len(claims) >= self.max_claims_per_paper:
                break
        
        doc.close()
        return claims
    
    def verify_claims(self, claims: List[Dict]) -> List[Dict]:
        """Verify claims using external sources"""
        
        for claim in claims:
            if self._should_verify(claim):
                verification = self.verifier.verify(claim['text'], {})
                claim['verification_status'] = 'Verified' if verification.verified else 'Unverified'
                claim['confidence_score'] = verification.confidence
                claim['explanation'] = verification.explanation
        
        return claims
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove citations
        text = re.sub(r'\[\d+\]|\(\w+ et al\., \d{4}\)', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _get_section_for_page(self, page_num: int, sections: Dict) -> str:
        """Get section for a page"""
        for section_name, section in sections.items():
            if hasattr(section, 'start_page') and hasattr(section, 'end_page'):
                if section.start_page <= page_num <= section.end_page:
                    return section_name
            elif isinstance(section, dict):
                if section.get('start', 0) <= page_num <= section.get('end', 999):
                    return section_name
        return "Unknown"
    
    def _should_verify(self, claim: Dict) -> bool:
        """Determine if a claim should be verified"""
        importance = claim.get('claim_score', 0)
        return importance > 0.5  # Only verify higher confidence claims
    
    def get_statistics(self, claims: List[Dict]) -> Dict:
        """Calculate claim statistics"""
        if not claims:
            return {}
        
        return {
            'total_claims': len(claims),
            'by_type': dict(Counter(c['claim_type'] for c in claims)),
            'avg_score': sum(c['claim_score'] for c in claims) / len(claims),
            'verified_count': sum(1 for c in claims if c.get('verification_status') == 'Verified')
        }


# Import fitz for page extraction
import fitz