"""Train a claim detection model"""

import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import os

# Configuration
MODEL_NAME = "bert-base-uncased"  # or "allenai/scibert_scivocab_uncased"
OUTPUT_DIR = "./models/claim_detector"
NUM_LABELS = 6  # 6 claim types
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5

# Claim type mapping
CLAIM_TYPES = [
    'Performance Comparison',  # 0
    'Novelty Claim',           # 1
    'Statistical Claim',       # 2
    'Research Gap',            # 3
    'Methodology Claim',       # 4
    'General Finding'          # 5
]

def load_data():
    """Load training data from CSV or SciFact"""
    
    # Try to load your training data
    csv_path = "data/annotations/minimal_training.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} examples from {csv_path}")
    else:
        # Create dummy data for testing
        print("⚠️ No training data found. Creating dummy data...")
        data = []
        for i in range(100):
            data.append({
                "text": f"This is example claim number {i}",
                "label": i % NUM_LABELS
            })
        df = pd.DataFrame(data)
    
    return df

def compute_metrics(eval_pred):
    """Calculate evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_weighted': f1_score(labels, predictions, average='weighted'),
    }

def tokenize_function(examples, tokenizer):
    """Tokenize text for BERT"""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

def train():
    """Main training function"""
    
    print("🚀 Starting training...")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Labels: {NUM_LABELS}")
    
    # Load data
    df = load_data()
    
    # Create dataset
    dataset = Dataset.from_pandas(df[['text', 'label']])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    # Load tokenizer and model
    print("📥 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True
    )
    
    # Tokenize datasets
    print("🔧 Tokenizing data...")
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    
    # Training arguments
    # Training arguments (updated for newer transformers)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",  # Changed from 'evaluation_strategy' to 'eval_strategy'
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        learning_rate=LEARNING_RATE,
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Train
    print("🏋️ Training...")
    trainer.train()
    
    # Save model
    print(f"💾 Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Evaluate
    print("📊 Final evaluation:")
    eval_results = trainer.evaluate(tokenized_datasets['test'])
    for key, value in eval_results.items():
        print(f"   {key}: {value:.4f}")
    
    print("✅ Training complete!")
    return trainer

if __name__ == "__main__":
    train()