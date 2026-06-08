"""Train a verifier model for claim verification (SUPPORT/CONTRADICT/NEI)"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import argparse
from .dataset import prepare_verifier_data


def compute_metrics(eval_pred):
    """Calculate evaluation metrics for 3-class classification"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'f1_weighted': f1_score(labels, predictions, average='weighted'),
    }


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize text for BERT input"""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def train_verifier(
    data_path: str = None,
    model_name: str = "allenai/scibert_scivocab_uncased",
    output_dir: str = "./models/verifier",
    num_labels: int = 3,  # SUPPORT, CONTRADICT, NEI
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5
):
    """Train a verifier model"""
    
    print("🚀 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    
    print("📊 Loading data...")
    dataset = prepare_verifier_data(data_path)
    
    # Tokenize
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    print("🏋️ Starting verifier training...")
    trainer.train()
    
    print("💾 Saving verifier model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluate on test set
    print("📈 Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets['test'])
    print(f"Test Results: {test_results}")
    
    # Detailed classification report
    print("\n📊 Detailed Classification Report:")
    predictions = trainer.predict(tokenized_datasets['test'])
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    target_names = ['SUPPORT', 'CONTRADICT', 'NEI']
    print(classification_report(true_labels, pred_labels, target_names=target_names))
    
    print(f"✅ Verifier training complete! Model saved to {output_dir}")
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to CSV with columns: text, label")
    parser.add_argument("--model_name", type=str, default="allenai/scibert_scivocab_uncased")
    parser.add_argument("--output_dir", type=str, default="./models/verifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    train_verifier(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )