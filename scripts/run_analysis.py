#!/usr/bin/env python3
"""Command-line script to run claim analysis on a PDF"""

import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.analyzer import ThesisClaimAnalyzer


def main():
    parser = argparse.ArgumentParser(description="Analyze claims in a research paper PDF")
    parser.add_argument("pdf_path", type=str, help="Path to PDF file")
    parser.add_argument("--output", "-o", type=str, default="analysis_results.xlsx", 
                       help="Output Excel file name")
    parser.add_argument("--use-ml", action="store_true", 
                       help="Use ML-enhanced detection (requires trained model)")
    parser.add_argument("--no-verify", action="store_true", 
                       help="Skip external verification")
    
    args = parser.parse_args()
    
    if not Path(args.pdf_path).exists():
        print(f"❌ File not found: {args.pdf_path}")
        sys.exit(1)
    
    # Initialize analyzer
    analyzer = ThesisClaimAnalyzer(use_ml=args.use_ml)
    
    # Run analysis
    analysis = analyzer.analyze_paper(args.pdf_path)
    
    # Export results
    analyzer.export_to_excel(analysis, args.output)
    
    print(f"\n✅ Done! Results saved to {args.output}")


if __name__ == "__main__":
    main()