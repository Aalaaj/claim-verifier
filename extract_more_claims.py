# extract_more_claims.py
"""Extract more candidate sentences from your PDFs"""

import fitz
import pandas as pd
import re
from pathlib import Path

def extract_all_sentences(pdf_path, max_sentences=200):
    """Extract all reasonable sentences from a PDF"""
    doc = fitz.open(pdf_path)
    sentences = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        raw_sentences = re.split(r'[.!?]+', text)
        
        for sent in raw_sentences:
            sent = sent.strip()
            # Filter: reasonable length, contains letters
            if 40 < len(sent) < 300 and re.search(r'[a-zA-Z]', sent):
                sentences.append({
                    'text': sent,
                    'page': page_num + 1,
                    'paper_name': Path(pdf_path).stem,
                    'is_claim': '',  # To fill manually
                    'claim_type': ''  # To fill manually
                })
            
            if len(sentences) >= max_sentences:
                break
        if len(sentences) >= max_sentences:
            break
    
    doc.close()
    return sentences

# Extract from all your PDFs
all_sentences = []
for pdf in Path("data/raw").glob("*.pdf"):
    print(f"Extracting from: {pdf.name}")
    sentences = extract_all_sentences(pdf, max_sentences=100)
    all_sentences.extend(sentences)

# Save for annotation
df = pd.DataFrame(all_sentences)
df.to_csv("data/annotations/to_annotate_round2.csv", index=False)
print(f"\n✅ Extracted {len(df)} sentences for annotation")
print("📝 Open this file in Excel and label is_claim (1/0) and claim_type")