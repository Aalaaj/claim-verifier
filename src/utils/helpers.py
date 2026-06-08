"""Helper utilities for the project"""

import json
import random
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from functools import wraps
import torch
import numpy as np


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Set up a logger with console handler"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def timer(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"⏱️ {func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper


def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file"""
    ensure_dir(filepath)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(filepath: str):
    """Create directory if it doesn't exist"""
    path = Path(filepath)
    if path.suffix:  # Has extension, get parent
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """Get available device (cuda/mps/cpu)"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def set_seed(seed: int = 42): # Makes results reproducible
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency-based method
    """
    # Remove stopwords (simple list)
    stopwords = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'the', 'this', 'these', 'those'
    }
    
    # Clean and split
    text = text.lower()
    words = re.findall(r'\b\w{3,}\b', text)
    
    # Filter stopwords
    keywords = [w for w in words if w not in stopwords]
    
    # Count frequencies
    freq = {}
    for w in keywords:
        freq[w] = freq.get(w, 0) + 1
    
    # Sort by frequency
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    return [kw for kw, _ in sorted_keywords[:top_n]]


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep letters, numbers, punctuation)
    text = re.sub(r'[^\w\s.,;:!?\-]', '', text)
    
    # Remove citation markers
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\([A-Za-z]+ et al\., \d{4}\)', '', text)
    
    return text.strip()


def chunk_text(text: str, max_length: int = 500) -> List[str]:
    """
    Split long text into chunks for processing
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) <= max_length:
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def format_claim_for_display(claim: Dict, max_length: int = 200) -> str:
    """Format a claim for readable display"""
    text = claim.get('text', '')[:max_length]
    if len(claim.get('text', '')) > max_length:
        text += '...'
    
    return (
        f"[{claim.get('claim_type', 'Unknown')}] "
        f"Score: {claim.get('claim_score', 0):.2f} | "
        f"Page: {claim.get('page', '?')} | "
        f"Section: {claim.get('section', 'Unknown')}\n"
        f"  \"{text}\"\n"
        f"  → {claim.get('explanation', 'No explanation')[:100]}"
    )


def merge_dicts(dict1: Dict, dict2: Dict, deep: bool = True) -> Dict:
    """Merge two dictionaries recursively"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if deep and key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value, deep)
        else:
            result[key] = value
    
    return result