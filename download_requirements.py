"""
Download all required files for offline training
Run this ONCE when you have good internet
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch

print("="*60)
print("📥 PRE-DOWNLOADING ALL REQUIREMENTS")
print("="*60)

# Create directories
os.makedirs("./models/pretrained", exist_ok=True)
os.makedirs("./data/cache", exist_ok=True)

# ============================================
# 1. Download the model
# ============================================
print("\n1️⃣ Downloading DeBERTa model (371MB)...")
print("   This may take 5-10 minutes...")

model_name = "microsoft/deberta-v3-base"

# Download tokenizer
print("   Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Download model
print("   Downloading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    ignore_mismatched_sizes=True
)

# Save locally for offline use
print("   Saving locally...")
model.save_pretrained("./models/pretrained/deberta-v3-base")
tokenizer.save_pretrained("./models/pretrained/deberta-v3-base")
print("   ✅ Model saved locally")

# ============================================
# 2. Download SciFact dataset (NEW METHOD)
# ============================================
print("\n2️⃣ Downloading SciFact dataset...")
print("   Using new Hugging Face dataset format...")

try:
    # NEW: Use the correct loading method
    from datasets import load_dataset
    
    # SciFact has two configurations: 'claims' and 'corpus'
    print("   Loading claims...")
    claims = load_dataset("scifact", "claims", trust_remote_code=False)
    
    print("   Loading corpus...")
    corpus = load_dataset("scifact", "corpus", trust_remote_code=False)
    
    # Save to local cache
    claims.save_to_disk("./data/cache/scifact_claims")
    corpus.save_to_disk("./data/cache/scifact_corpus")
    
    print(f"   ✅ Train claims: {len(claims['train'])}")
    print(f"   ✅ Validation claims: {len(claims['validation'])}")
    print(f"   ✅ Test claims: {len(claims['test'])}")
    print(f"   ✅ Corpus documents: {len(corpus['train'])}")
    
except Exception as e:
    print(f"   ⚠️ Method 1 failed: {e}")
    print("   Trying alternative method...")
    
    # Alternative: Load without splitting
    try:
        claims = load_dataset("scifact", trust_remote_code=True)
        print("   ✅ Loaded with alternative method")
    except Exception as e2:
        print(f"   ❌ Both methods failed: {e2}")
        print("\n   📥 Downloading from GitHub as fallback...")
        
        # Fallback: Download raw JSONL files
        import requests
        
        files = {
            "claims_train.jsonl": "https://raw.githubusercontent.com/allenai/scifact/master/data/claims_train.jsonl",
            "claims_dev.jsonl": "https://raw.githubusercontent.com/allenai/scifact/master/data/claims_dev.jsonl",
            "corpus.jsonl": "https://raw.githubusercontent.com/allenai/scifact/master/data/corpus.jsonl"
        }
        
        for filename, url in files.items():
            print(f"   Downloading {filename}...")
            response = requests.get(url)
            with open(f"./data/cache/{filename}", "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"   ✅ Saved {filename}")
        
        print("   ✅ Fallback download complete")

print("   ✅ Dataset cached locally")

# ============================================
# 3. Test offline loading
# ============================================
print("\n3️⃣ Testing offline loading...")

# Set offline mode
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

try:
    # Try loading model from local disk
    tokenizer = AutoTokenizer.from_pretrained("./models/pretrained/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "./models/pretrained/deberta-v3-base"
    )
    print("   ✅ Model loads from local disk")
    
    # Try loading dataset from cache
    from datasets import load_from_disk
    if os.path.exists("./data/cache/scifact_claims"):
        claims = load_from_disk("./data/cache/scifact_claims")
        print("   ✅ Dataset loads from local cache")
    else:
        print("   ⚠️ Dataset cache not found, but raw files available")
    
except Exception as e:
    print(f"   ❌ Offline loading failed: {e}")

print("\n" + "="*60)
print("✅ ALL REQUIREMENTS DOWNLOADED!")
print("="*60)
print("\n💡 You can now:")
print("   1. Disconnect from the internet")
print("   2. Run the training script")
print("   3. Training will work completely offline")