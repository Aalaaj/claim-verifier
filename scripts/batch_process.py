"""Batch process multiple PDF files"""

import argparse
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.analyzer import ThesisClaimAnalyzer


def batch_process(input_dir: str, output_dir: str, use_ml: bool = False):
    """Process all PDFs in a directory"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {input_dir}")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files")
    
    analyzer = ThesisClaimAnalyzer(use_ml=use_ml)
    
    results = []
    for pdf_path in tqdm(pdf_files, desc="Processing"):
        try:
            output_file = output_path / f"{pdf_path.stem}_analysis.xlsx"
            analysis = analyzer.analyze_paper(str(pdf_path), verbose=False)
            analyzer.export_to_excel(analysis, str(output_file))
            results.append({
                'file': pdf_path.name,
                'status': 'success',
                'claims': analysis.overall_statistics.get('total_claims', 0)
            })
        except Exception as e:
            results.append({
                'file': pdf_path.name,
                'status': 'failed',
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*50)
    print("BATCH PROCESSING SUMMARY")
    print("="*50)
    success = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"✅ Successful: {len(success)}")
    print(f"❌ Failed: {len(failed)}")
    
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  - {f['file']}: {f.get('error', 'Unknown error')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Directory containing PDFs")
    parser.add_argument("--output", "-o", type=str, default="./results", help="Output directory")
    parser.add_argument("--use-ml", action="store_true", help="Use ML detection")
    args = parser.parse_args()
    
    batch_process(args.input_dir, args.output, args.use_ml)