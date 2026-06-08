# Thesis Claim Analyzer

A framework for detecting, extracting, and analyzing claims in research papers.

## Features

- 📄 PDF parsing and section detection
- 🔍 Claim extraction using keyword and ML-based methods
- 📊 Trust score calculation with mathematical formulas
- 🔗 External verification via Semantic Scholar API
- 📈 Excel export with detailed results

## Installation

```bash
git clone https://github.com/
cd thesis-claim-analyzer
pip install -r requirements.txt
python -m spacy download en_core_web_sm



venv\Scripts\activate 

py scripts/run_analysis.py data\raw\testThesis.pdf --output results.xlsx  