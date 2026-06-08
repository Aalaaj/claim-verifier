"""Main ThesisClaimAnalyzer class - orchestrates the entire pipeline"""

import fitz
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

from src.models.claim_detector import KeywordClaimDetector, BertClaimDetector, EnsembleDetector
from src.models.verifier import SemanticScholarVerifier
from src.models.trust_metrics import TrustMetricsCalculator, TrustMetrics
from src.utils.export import ExcelExporter


@dataclass
class Claim:
    """Represents a single claim extracted from the paper"""
    text: str
    page: int
    section: str
    claim_type: str
    claim_score: float
    importance: str
    structural_strength: str
    is_fact: bool
    entities: List[Tuple[str, str]]
    verification_status: str = "Pending"
    confidence_score: float = 0.0
    explanation: str = ""
    relevance_score: float = 0.0
    trust_score: float = 0.0
    justification: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PaperAnalysis:
    """Complete analysis results for a paper"""
    paper_path: str
    paper_title: str
    abstract: str
    analysis_timestamp: str
    claims: List[Claim]
    sections_found: Dict[str, Any]
    reference_issues: List[Dict]
    overall_statistics: Dict[str, Any]
    config_used: Dict[str, Any]


class ThesisClaimAnalyzer:
    """Main orchestrator for claim analysis"""
    
    def __init__(self, use_ml: bool = False, config: Optional[Dict] = None):
        # Setup detectors
        keyword_detector = KeywordClaimDetector()
        
        if use_ml:
            bert_detector = BertClaimDetector()
            self.claim_detector = EnsembleDetector([keyword_detector, bert_detector])
        else:
            self.claim_detector = keyword_detector
        
        self.verifier = SemanticScholarVerifier()
        self.trust_calculator = TrustMetricsCalculator()
        self.exporter = ExcelExporter()
        
        self.config = {
            'min_claim_score': 0.3,
            'min_claim_length': 40,
            'max_claims_per_paper': 500,
            'verify_only_important': True,
            'min_importance_for_verification': 'Medium',
            'check_relevance': True,
            'use_ml': use_ml
        }
        if config:
            self.config.update(config)
        
        # Load spaCy
        import spacy
        self.nlp = spacy.load("en_core_web_sm")
        
        print(f"✅ ThesisClaimAnalyzer initialized")
        print(f"   ML Enabled: {self.config['use_ml']}")
    
    def analyze_paper(self, pdf_path: str, verbose: bool = True) -> PaperAnalysis:
        """Main analysis pipeline"""
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"📄 Analyzing: {Path(pdf_path).name}")
            print(f"{'='*80}")
        
        # Extract paper structure
        doc = fitz.open(pdf_path)
        sections = self._identify_sections(doc)
        title, abstract = self._extract_title_abstract(doc)
        
        # Extract claims
        claims = self._extract_claims(doc, sections)
        
        # Verify claims
        for claim in claims:
            if self._should_verify(claim):
                verification = self.verifier.verify(claim.text, {})
                claim.verification_status = 'Verified' if verification.verified else 'Unverified'
                claim.confidence_score = verification.confidence
                claim.explanation = verification.explanation
        
        # Calculate trust metrics
        for claim in claims:
            trust = self._calculate_trust_metrics(claim)
            claim.trust_score = trust.overall_score
            claim.justification = trust.explanation
        
        # Calculate statistics
        statistics = self._calculate_statistics(claims)
        
        # Create analysis object
        analysis = PaperAnalysis(
            paper_path=pdf_path,
            paper_title=title,
            abstract=abstract[:500] + "..." if len(abstract) > 500 else abstract,
            analysis_timestamp=datetime.now().isoformat(),
            claims=claims,
            sections_found=sections,
            reference_issues=[],
            overall_statistics=statistics,
            config_used=self.config
        )
        
        if verbose:
            self._print_summary(analysis)
        
        doc.close()
        return analysis
    
    def _identify_sections(self, doc) -> Dict:
        """Identify section boundaries in PDF"""
        section_headers = [
            "abstract", "introduction", "related work", "methodology",
            "methods", "results", "discussion", "conclusion", "references"
        ]
        
        section_starts = {}
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().lower()
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                for header in section_headers:
                    if header in line and len(line) < 50:
                        if header not in section_starts:
                            section_starts[header] = page_num + 1
                        break
        
        boundaries = {}
        sorted_sections = sorted(section_starts.items(), key=lambda x: x[1])
        
        for i, (section, start_page) in enumerate(sorted_sections):
            end_page = sorted_sections[i + 1][1] - 1 if i + 1 < len(sorted_sections) else len(doc)
            boundaries[section] = {'start': start_page, 'end': end_page}
        
        return boundaries
    
    def _extract_title_abstract(self, doc) -> Tuple[str, str]:
        """Extract title and abstract from first page"""
        first_page = doc[0].get_text()
        lines = first_page.split('\n')
        
        title = ""
        abstract = ""
        
        for i, line in enumerate(lines):
            line = line.strip()
            if 'abstract' in line.lower():
                abstract_start = i + 1
                abstract_lines = []
                for line in lines[abstract_start:]:
                    if line.strip() and len(abstract_lines) < 20:
                        abstract_lines.append(line.strip())
                    elif 'introduction' in line.lower() and len(abstract_lines) > 0:
                        break
                abstract = ' '.join(abstract_lines)
                break
            elif not title and line and len(line) > 10:
                title = line
        
        return title, abstract
    
    def _extract_claims(self, doc, sections: Dict) -> List[Claim]:
        """Extract claims from document"""
        claims = []
        seen_texts = set()
        
        for page_num in range(len(doc)):
            section = self._get_section_for_page(page_num + 1, sections)
            page = doc.load_page(page_num)
            blocks = page.get_text("blocks")
            
            for block in blocks:
                text = block[4].strip()
                
                if len(text) < self.config['min_claim_length'] or text in seen_texts:
                    continue
                
                # Clean text
                text = re.sub(r'\[\d+\]|\(\w+ et al\., \d{4}\)', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                # Detect claim
                claim_type, claim_score = self.claim_detector.detect_claims(text, {'section': section})
                
                if claim_score >= self.config['min_claim_score']:
                    doc_nlp = self.nlp(text)
                    entities = [(ent.text, ent.label_) for ent in doc_nlp.ents]
                    importance = self._calculate_importance(text, section, claim_score, len(entities))
                    
                    claim = Claim(
                        text=text,
                        page=page_num + 1,
                        section=section.capitalize() if section != "Unknown" else "Unknown",
                        claim_type=claim_type,
                        claim_score=round(claim_score, 2),
                        importance=importance,
                        structural_strength=self._calculate_structural_strength(claim_type, section, text),
                        is_fact=self._is_fact(text),
                        entities=entities
                    )
                    
                    claims.append(claim)
                    seen_texts.add(text)
                    
                    if len(claims) >= self.config['max_claims_per_paper']:
                        break
            
            if len(claims) >= self.config['max_claims_per_paper']:
                break
        
        return claims
    
    def _get_section_for_page(self, page_num: int, sections: Dict) -> str:
        """Get section name for a given page number"""
        for section, boundaries in sections.items():
            if boundaries['start'] <= page_num <= boundaries['end']:
                return section
        return "Unknown"
    
    def _is_fact(self, text: str) -> bool:
        """Check if a sentence is likely a factual statement"""
        fact_indicators = [
            'is defined as', 'refers to', 'is a', 'are the', 'can be',
            'typically', 'usually', 'generally', 'commonly', 'often'
        ]
        return any(fi in text.lower() for fi in fact_indicators)
    
    def _calculate_importance(self, text: str, section: str, claim_score: float, num_entities: int) -> str:
        """Calculate claim importance level"""
        importance = claim_score * 0.4
        
        section_weights = {
            'abstract': 1.0, 'conclusion': 1.0, 'results': 0.9,
            'discussion': 0.8, 'introduction': 0.6, 'methodology': 0.5
        }
        
        section_weight = section_weights.get(section.lower(), 0.3)
        importance += section_weight * 0.3
        importance += min(num_entities * 0.1, 0.2)
        importance = min(importance, 1.0)
        
        if importance > 0.8:
            return 'Critical'
        elif importance > 0.6:
            return 'High'
        elif importance > 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_structural_strength(self, category: str, section: str, sentence: str) -> str:
        """Calculate structural strength as string (High/Medium/Low)"""
        ss_score = self.trust_calculator.calculate_structural_strength(category, section, sentence)
        
        if ss_score >= 0.8:
            return "High"
        elif ss_score >= 0.5:
            return "Medium"
        else:
            return "Low"
    
    def _should_verify(self, claim: Claim) -> bool:
        """Determine if a claim should be verified externally"""
        if not self.config['verify_only_important']:
            return True
        
        importance_order = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        min_importance = self.config['min_importance_for_verification']
        
        return importance_order[claim.importance] >= importance_order[min_importance]
    
    def _calculate_trust_metrics(self, claim: Claim) -> TrustMetrics:
        """Calculate all trust metrics for a claim"""
        
        # Calculate components
        s_int = self.trust_calculator.calculate_internal_consistency(claim.claim_score)
        
        # Simplified external verification (would use real citation data in production)
        s_ext = self.trust_calculator.calculate_external_verification(0, 0)
        
        s_cit = 0.3  # Placeholder
        s_meth = self.trust_calculator.calculate_methodological_quality(claim.text)
        s_rep = self.trust_calculator.calculate_reproducibility(claim.text)
        
        # Weights
        weights = {
            'internal_consistency': 0.15,
            'external_verification': 0.35,
            'citation_quality': 0.20,
            'methodological_soundness': 0.20,
            'reproducibility': 0.10
        }
        
        metrics = {
            'internal_consistency': s_int,
            'external_verification': s_ext,
            'citation_quality': s_cit,
            'methodological_soundness': s_meth,
            'reproducibility': s_rep
        }
        
        overall = self.trust_calculator.calculate_overall_trust(metrics, weights)
        
        explanation = f"{claim.claim_type} claim with {claim.importance.lower()} importance. Trust: {overall:.0%}"
        
        return TrustMetrics(
            overall_score=overall,
            internal_consistency=s_int,
            external_verification=s_ext,
            citation_quality=s_cit,
            methodological_soundness=s_meth,
            reproducibility_potential=s_rep,
            explanation=explanation
        )
    
    def _calculate_statistics(self, claims: List[Claim]) -> Dict:
        """Calculate overall statistics"""
        if not claims:
            return {}
        
        return {
            'total_claims': len(claims),
            'by_type': dict(Counter(c.claim_type for c in claims)),
            'by_importance': dict(Counter(c.importance for c in claims)),
            'avg_claim_score': np.mean([c.claim_score for c in claims]),
            'avg_trust_score': np.mean([c.trust_score for c in claims]),
            'verified_count': sum(1 for c in claims if c.verification_status == 'Verified'),
            'high_trust_count': sum(1 for c in claims if c.trust_score > 0.7),
            'critical_claims': sum(1 for c in claims if c.importance == 'Critical')
        }
    
    def _print_summary(self, analysis: PaperAnalysis):
        """Print analysis summary"""
        print(f"\n{'='*80}")
        print("📊 ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        stats = analysis.overall_statistics
        
        print(f"\n📄 Paper: {analysis.paper_title[:80]}...")
        print(f"\n📈 Claims:")
        print(f"   Total: {stats.get('total_claims', 0)}")
        print(f"   Critical: {stats.get('critical_claims', 0)}")
        print(f"   Verified: {stats.get('verified_count', 0)}")
        
        print(f"\n   Claim Types:")
        for ctype, count in sorted(stats.get('by_type', {}).items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     - {ctype}: {count}")
        
        print(f"\n🔒 Trust:")
        print(f"   Average Score: {stats.get('avg_trust_score', 0):.2f}")
        print(f"   High Trust: {stats.get('high_trust_count', 0)}")
    
    def export_to_excel(self, analysis: PaperAnalysis, filename: str):
        """Export analysis to Excel"""
        self.exporter.export(analysis, filename)