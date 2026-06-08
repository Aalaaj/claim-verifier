"""Dataset preparation for training"""

import pandas as pd
import json
from datasets import Dataset, DatasetDict
from typing import Tuple, Optional
import requests


def download_scifact() -> Tuple[list, list]: #Downloads SciFact dataset
    """Download SciFact dataset from GitHub"""
    train_url = "https://raw.githubusercontent.com/allenai/scifact/master/data/claims_train.jsonl"
    dev_url = "https://raw.githubusercontent.com/allenai/scifact/master/data/claims_dev.jsonl"
    
    def load_jsonl(url):
        response = requests.get(url)
        return [json.loads(line) for line in response.text.strip().split('\n') if line]
    
    train_claims = load_jsonl(train_url)
    dev_claims = load_jsonl(dev_url)
    
    print(f"Downloaded {len(train_claims)} training claims")
    print(f"Downloaded {len(dev_claims)} dev claims")
    
    return train_claims, dev_claims


def prepare_claim_detection_data(csv_path: Optional[str] = None) -> DatasetDict: # Formats data for training
    """
    Prepare data for claim detection (6 claim types)
    
    Expected CSV columns: text, label
    label: 0-5 mapping to claim types
    """
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        # Load SciFact and adapt labels (simplified)
        train_claims, dev_claims = download_scifact()
        
        # Convert to DataFrame (simplified - adjust as needed)
        train_data = [{"text": c["claim"], "label": 0} for c in train_claims]  # Placeholder labels
        dev_data = [{"text": c["claim"], "label": 0} for c in dev_claims]
        
        df = pd.DataFrame(train_data + dev_data)
    
    dataset = Dataset.from_pandas(df[['text', 'label']])
    
    # Split into train/val/test (80/10/10)
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    test_val = train_test['test'].train_test_split(test_size=0.5, seed=42)
    
    return DatasetDict({
        'train': train_test['train'],
        'validation': test_val['train'],
        'test': test_val['test']
    })


def prepare_verifier_data(csv_path: Optional[str] = None) -> DatasetDict:
    """
    Prepare data for verifier (SUPPORT/CONTRADICT/NEI)
    
    Expected CSV columns: claim_text, label (0=SUPPORT, 1=CONTRADICT, 2=NEI)
    """
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        # Load SciFact with labels
        train_claims, dev_claims = download_scifact()
        
        label_map = {'SUPPORT': 0, 'CONTRADICT': 1, 'NEI': 2}
        
        train_data = [{"text": c["claim"], "label": label_map[c.get("label", "NEI")]} 
                     for c in train_claims]
        dev_data = [{"text": c["claim"], "label": label_map[c.get("label", "NEI")]} 
                   for c in dev_claims]
        
        df = pd.DataFrame(train_data + dev_data)
    
    dataset = Dataset.from_pandas(df[['text', 'label']])
    
    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    test_val = train_test['test'].train_test_split(test_size=0.5, seed=42)
    
    return DatasetDict({
        'train': train_test['train'],
        'validation': test_val['train'],
        'test': test_val['test']
    })