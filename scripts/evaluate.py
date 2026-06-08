"""
Evaluate the keyword-based claim detector (baseline for Paper 1)
No ML required - just your current system
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.analyzer import ThesisClaimAnalyzer
from src.models.claim_detector import KeywordClaimDetector


class BaselineEvaluator:
    """Evaluate keyword-based claim detector"""
    
    def __init__(self):
        self.detector = KeywordClaimDetector()
        self.claim_types = [
            'Performance Comparison',
            'Novelty Claim', 
            'Statistical Claim',
            'Research Gap',
            'Methodology Claim',
            'General Finding'
        ]
    
    def load_ground_truth(self, csv_path):
        """Load annotated ground truth"""
        df = pd.read_csv(csv_path)
        required_cols = ['text', 'is_claim', 'claim_type']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        return df
    
    def evaluate_claim_detection(self, ground_truth_df):
        """
        Evaluate binary claim detection (is this a claim or not?)
        """
        print("\n" + "="*60)
        print("📊 CLAIM DETECTION EVALUATION")
        print("="*60)
        
        y_true = []
        y_pred = []
        
        for _, row in ground_truth_df.iterrows():
            true_label = row['is_claim']
            
            # Run keyword detector
            claim_type, score = self.detector.detect_claims(row['text'], {})
            
            # If score > 0, consider it a detected claim
            pred_label = 1 if score > 0 else 0
            
            y_true.append(true_label)
            y_pred.append(pred_label)
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        print(f"\n📈 Detection Metrics:")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall:    {recall:.2%}")
        print(f"   F1 Score:  {f1:.2%}")
        print(f"   Accuracy:  {accuracy:.2%}")
        
        print(f"\n📊 Confusion Matrix:")
        print(f"   True Positives:  {tp}  (correctly identified claims)")
        print(f"   False Positives: {fp}  (false alarms)")
        print(f"   False Negatives: {fn}  (missed claims)")
        print(f"   True Negatives:  {tn}  (correctly ignored)")
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    def evaluate_claim_typing(self, ground_truth_df):
        """
        Evaluate claim type classification (only on actual claims)
        """
        print("\n" + "="*60)
        print("📊 CLAIM TYPING EVALUATION")
        print("="*60)
        
        y_true = []
        y_pred = []
        skipped = 0
        
        for _, row in ground_truth_df.iterrows():
            # Only evaluate actual claims
            if row['is_claim'] != 1:
                continue
            
            true_type = row['claim_type']
            if pd.isna(true_type) or true_type == '':
                skipped += 1
                continue
            
            # Run keyword detector
            pred_type, score = self.detector.detect_claims(row['text'], {})
            
            # If detector returned "No Claim", treat as General Finding
            if pred_type == "No Claim":
                pred_type = "General Finding"
            
            y_true.append(true_type)
            y_pred.append(pred_type)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-type metrics
        print(f"\n📈 Typing Metrics (on {len(y_true)} claims, skipped {skipped}):")
        print(f"   Accuracy: {accuracy:.2%}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, labels=self.claim_types))
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
    
    def evaluate_on_papers(self, pdf_paths, ground_truth_df):
        """
        Evaluate on full papers (end-to-end)
        """
        print("\n" + "="*60)
        print("📊 END-TO-END PAPER EVALUATION")
        print("="*60)
        
        analyzer = ThesisClaimAnalyzer(use_ml=False)
        
        all_results = []
        
        for pdf_path in pdf_paths:
            print(f"\n📄 Analyzing: {Path(pdf_path).name}")
            
            # Run analysis
            analysis = analyzer.analyze_paper(pdf_path, verbose=False)
            
            # Get detected claims
            detected_claims = {claim.text: claim for claim in analysis.claims}
            
            # Compare with ground truth for this paper
            paper_gt = ground_truth_df[ground_truth_df['paper_name'] == Path(pdf_path).stem]
            
            tp = 0  # correctly detected
            fp = 0  # false positives
            fn = 0  # missed claims
            
            for _, row in paper_gt.iterrows():
                if row['is_claim'] == 1:
                    if row['text'] in detected_claims:
                        tp += 1
                    else:
                        fn += 1
            
            # False positives: detected but not in ground truth (simplified)
            for claim_text in detected_claims:
                if claim_text not in paper_gt['text'].values:
                    fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            all_results.append({
                'paper': Path(pdf_path).name,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            print(f"   TP: {tp}, FP: {fp}, FN: {fn}")
            print(f"   Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")
        
        # Average across papers
        avg_precision = np.mean([r['precision'] for r in all_results])
        avg_recall = np.mean([r['recall'] for r in all_results])
        avg_f1 = np.mean([r['f1'] for r in all_results])
        
        print(f"\n📊 AVERAGE ACROSS {len(all_results)} PAPERS:")
        print(f"   Precision: {avg_precision:.2%}")
        print(f"   Recall: {avg_recall:.2%}")
        print(f"   Recall: {avg_recall:.2%}")
        print(f"   F1 Score: {avg_f1:.2%}")
        
        return all_results


def main():
    print("="*60)
    print("🔬 BASELINE EVALUATION (Paper 1)")
    print("="*60)
    print("Evaluating keyword-based claim detector")
    print("No ML - just your current system")
    
    evaluator = BaselineEvaluator()
    
    # Load ground truth
    gt_path = "data/annotations/ground_truth.csv"
    if not os.path.exists(gt_path):
        print(f"\n❌ Ground truth file not found: {gt_path}")
        print("\nPlease create a CSV file with columns:")
        print("  - text: the sentence to evaluate")
        print("  - is_claim: 1 if claim, 0 if not")
        print("  - claim_type: type of claim (if is_claim=1)")
        print("  - paper_name: identifier for the paper")
        return
    
    ground_truth = evaluator.load_ground_truth(gt_path)
    print(f"\n✅ Loaded {len(ground_truth)} annotated sentences")
    
    # 1. Evaluate claim detection
    detection_results = evaluator.evaluate_claim_detection(ground_truth)
    
    # 2. Evaluate claim typing
    typing_results = evaluator.evaluate_claim_typing(ground_truth)
    
    # 3. Evaluate on full papers (if you have PDFs)
    pdf_dir = "data/raw"
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    
    if pdf_files and 'paper_name' in ground_truth.columns:
        paper_results = evaluator.evaluate_on_papers(pdf_files, ground_truth)
    else:
        print("\n⚠️ Skipping end-to-end evaluation (no PDFs or paper_name column)")
    
    # Save results
    results = {
        'detection': detection_results,
        'typing': typing_results,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    import json
    with open('baseline_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n✅ Results saved to: baseline_evaluation_results.json")
    print("\n💡 These are your baseline metrics for Paper 1")
    print("   After adding ML, compare to show improvement")


if __name__ == "__main__":
    main()