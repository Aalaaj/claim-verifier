"""Streamlit UI for Thesis Claim Analyzer with Evaluation Dashboard"""

import streamlit as st
import pandas as pd
import tempfile
import os
import json
from pathlib import Path
from datetime import datetime

# Import your analyzer
from src.pipeline.analyzer import ThesisClaimAnalyzer
from src.models.claim_detector import KeywordClaimDetector

# Page configuration
st.set_page_config(
    page_title="Thesis Claim Analyzer",
    page_icon="📄",
    layout="wide"
)

# ============================================
# CACHED FUNCTIONS
# ============================================

@st.cache_resource
def get_analyzer(use_ml):
    return ThesisClaimAnalyzer(use_ml=use_ml)

@st.cache_resource
def get_keyword_detector():
    return KeywordClaimDetector()

@st.cache_data
def load_ground_truth():
    """Load ground truth if it exists"""
    gt_path = "data/annotations/ground_truth.csv"
    if os.path.exists(gt_path):
        return pd.read_csv(gt_path)
    return None

# ============================================
# EVALUATION FUNCTIONS
# ============================================

def evaluate_detection(detector, ground_truth_df):
    """Evaluate claim detection performance"""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    y_true = []
    y_pred = []
    
    for _, row in ground_truth_df.iterrows():
        if pd.isna(row.get('is_claim')):
            continue
            
        true_label = row['is_claim']
        claim_type, score = detector.detect_claims(row['text'], {})
        pred_label = 1 if score > 0 else 0
        
        y_true.append(true_label)
        y_pred.append(pred_label)
    
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'total_samples': len(y_true),
        'true_positives': sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1),
        'false_positives': sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1),
        'false_negatives': sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0),
        'true_negatives': sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0),
    }
    
    return metrics

def evaluate_typing(detector, ground_truth_df):
    """Evaluate claim type classification"""
    from sklearn.metrics import accuracy_score
    
    claim_types = ['Performance Comparison', 'Novelty Claim', 'Statistical Claim', 
                   'Research Gap', 'Methodology Claim', 'General Finding']
    
    y_true = []
    y_pred = []
    
    for _, row in ground_truth_df.iterrows():
        if row.get('is_claim') != 1 or pd.isna(row.get('claim_type')):
            continue
            
        true_type = row['claim_type']
        pred_type, _ = detector.detect_claims(row['text'], {})
        
        if pred_type == "No Claim":
            pred_type = "General Finding"
        
        y_true.append(true_type)
        y_pred.append(pred_type)
    
    accuracy = accuracy_score(y_true, y_pred) if y_true else 0
    
    from collections import defaultdict
    per_type_correct = defaultdict(int)
    per_type_total = defaultdict(int)
    
    for true, pred in zip(y_true, y_pred):
        per_type_total[true] += 1
        if true == pred:
            per_type_correct[true] += 1
    
    per_type_accuracy = {
        ct: per_type_correct.get(ct, 0) / per_type_total.get(ct, 1)
        for ct in claim_types
        if per_type_total.get(ct, 0) > 0
    }
    
    return {
        'accuracy': accuracy,
        'total_claims': len(y_true),
        'per_type_accuracy': per_type_accuracy
    }

# ============================================
# TABS: ANALYSIS + EVALUATION
# ============================================

tab1, tab2 = st.tabs(["🔍 Analyze Paper", "📊 Model Evaluation"])

# ============================================
# TAB 1: ANALYZE PAPER (YOUR ORIGINAL CODE)
# ============================================

with tab1:
    # Title
    st.markdown("### Detect and analyze claims in research papers")
    
    # Sidebar for settings (moved to main area for tab1)
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**⚙️ Settings**")
        use_ml = st.checkbox("Use ML detection", value=False, 
                             help="Requires trained model. Currently using keyword detection.", key="use_ml_tab1")
        verify = st.checkbox("Verify claims externally", value=True, key="verify_tab1")
    
    with col2:
        st.info("""
        **About:** This tool extracts claims from research papers and calculates trust scores based on
        internal consistency, external verification, citation quality, methodological soundness, and reproducibility.
        """)
    
    # Main area
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        
        # Analyze button
        if st.button("🔍 Analyze Paper", type="primary"):
            with st.spinner("Analyzing paper... This may take a moment."):
                analyzer = get_analyzer(use_ml)
                analysis = analyzer.analyze_paper(tmp_path)
            
            # Display results
            st.header("📊 Analysis Results")
            
            # Statistics in columns
            stats = analysis.overall_statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Claims", stats.get('total_claims', 0))
            with col2:
                st.metric("Critical Claims", stats.get('critical_claims', 0))
            with col3:
                st.metric("Verified Claims", stats.get('verified_count', 0))
            with col4:
                avg_trust = stats.get('avg_trust_score', 0)
                st.metric("Avg Trust Score", f"{avg_trust:.2f}")
            
            # Claim types chart
            st.subheader("📈 Claims by Type")
            if stats.get('by_type'):
                type_df = pd.DataFrame(list(stats['by_type'].items()), 
                                       columns=['Type', 'Count'])
                st.bar_chart(type_df.set_index('Type'))
            
            # Claims table
            st.subheader("📋 Extracted Claims")
            claims_data = []
            for claim in analysis.claims:
                claims_data.append({
                    'Type': claim.claim_type,
                    'Trust Score': f"{claim.trust_score:.0%}",
                    'Importance': claim.importance,
                    'Text': claim.text[:200] + "..." if len(claim.text) > 200 else claim.text,
                    'Section': claim.section,
                    'Page': claim.page
                })
            
            claims_df = pd.DataFrame(claims_data)
            st.dataframe(claims_df, use_container_width=True)
            
            # Export button
            output_file = f"{Path(uploaded_file.name).stem}_results.xlsx"
            analyzer.export_to_excel(analysis, output_file)
            
            with open(output_file, "rb") as f:
                st.download_button(
                    label="📥 Download Results (Excel)",
                    data=f,
                    file_name=output_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Cleanup
            os.unlink(tmp_path)
            if os.path.exists(output_file):
                os.unlink(output_file)

# ============================================
# TAB 2: MODEL EVALUATION (NEW)
# ============================================

with tab2:
    st.markdown("### 📊 Model Performance Evaluation")
    st.markdown("Evaluate the keyword-based claim detector against ground truth data")
    
    # Load ground truth
    ground_truth = load_ground_truth()
    
    if ground_truth is None:
        st.warning("⚠️ No ground truth found! Please create `data/annotations/ground_truth.csv`")
        
        with st.expander("📝 How to create ground truth"):
            st.markdown("""
            Create a CSV file at `data/annotations/ground_truth.csv` with these columns:
            
            | text | is_claim | claim_type | paper_name | confidence |
            |------|----------|------------|------------|------------|
            | Our model achieves 95% accuracy | 1 | Performance Comparison | paper1 | 3 |
            | This paper is organized as follows | 0 | | paper1 | 3 |
            
            **Columns:**
            - `text`: The sentence to evaluate
            - `is_claim`: 1 if claim, 0 if not
            - `claim_type`: One of: Performance Comparison, Novelty Claim, Statistical Claim, Research Gap, Methodology Claim, General Finding
            - `paper_name`: Identifier for the paper (optional)
            - `confidence`: 1-3 (how sure you are)
            """)
            
            # Download template button
            template = pd.DataFrame({
                'text': ['Example claim sentence', 'Example non-claim sentence'],
                'is_claim': [1, 0],
                'claim_type': ['Performance Comparison', ''],
                'paper_name': ['paper1', 'paper1'],
                'confidence': [3, 3]
            })
            
            csv = template.to_csv(index=False)
            st.download_button("📥 Download Template CSV", csv, "ground_truth_template.csv", "text/csv")
    else:
        st.success(f"✅ Loaded {len(ground_truth)} annotated sentences")
        
        # Show ground truth summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sentences", len(ground_truth))
        with col2:
            num_claims = len(ground_truth[ground_truth['is_claim'] == 1])
            st.metric("Claims", num_claims)
        with col3:
            num_non_claims = len(ground_truth[ground_truth['is_claim'] == 0])
            st.metric("Non-Claims", num_non_claims)
        
        # Run evaluation
        detector = get_keyword_detector()
        
        with st.spinner("Evaluating model on ground truth..."):
            detection_metrics = evaluate_detection(detector, ground_truth)
            typing_metrics = evaluate_typing(detector, ground_truth)
        
        # ====================================
        # CLAIM DETECTION RESULTS
        # ====================================
        st.subheader("🎯 Claim Detection Results")
        st.markdown("*Does the system correctly identify which sentences are claims?*")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Precision", f"{detection_metrics['precision']:.1%}")
        with col2:
            st.metric("Recall", f"{detection_metrics['recall']:.1%}")
        with col3:
            st.metric("F1 Score", f"{detection_metrics['f1']:.1%}")
        with col4:
            st.metric("Accuracy", f"{detection_metrics['accuracy']:.1%}")
        
        # Confusion Matrix as DataFrame
        st.subheader("📊 Confusion Matrix")
        
        cm_df = pd.DataFrame({
            '': ['Actual: Non-Claim', 'Actual: Claim'],
            'Predicted: Non-Claim': [detection_metrics['true_negatives'], detection_metrics['false_negatives']],
            'Predicted: Claim': [detection_metrics['false_positives'], detection_metrics['true_positives']]
        })
        st.dataframe(cm_df, use_container_width=True, hide_index=True)
        
        # Interpretation based on F1 score
        if detection_metrics['f1'] < 0.4:
            st.info("📉 **Baseline Performance**: Your keyword detector is establishing a lower bound. This is expected for Paper 1. After adding ML, you will show improvement.")
        elif detection_metrics['f1'] < 0.6:
            st.success("📈 **Moderate Performance**: Your detector is working reasonably well for a keyword baseline.")
        else:
            st.success("🎯 **Strong Performance**: Your detector is performing well!")
        
        # ====================================
        # CLAIM TYPING RESULTS
        # ====================================
        if typing_metrics['total_claims'] > 0:
            st.subheader("🏷️ Claim Type Classification Results")
            st.markdown("*Does the system correctly classify the type of each claim?*")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Typing Accuracy", f"{typing_metrics['accuracy']:.1%}")
            with col2:
                st.metric("Total Claims Evaluated", typing_metrics['total_claims'])
            
            # Per-type accuracy table
            if typing_metrics['per_type_accuracy']:
                st.subheader("Accuracy by Claim Type")
                type_acc_df = pd.DataFrame([
                    {'Claim Type': ct, 'Accuracy': f"{acc:.1%}"}
                    for ct, acc in typing_metrics['per_type_accuracy'].items()
                ])
                st.dataframe(type_acc_df, use_container_width=True, hide_index=True)
        
        # ====================================
        # EXPORT RESULTS
        # ====================================
        st.subheader("💾 Export Results")
        
        export_results = {
            'detection_metrics': {k: v for k, v in detection_metrics.items() 
                                  if k not in ['y_true', 'y_pred']},
            'typing_accuracy': typing_metrics['accuracy'],
            'total_samples': detection_metrics['total_samples'],
            'timestamp': datetime.now().isoformat()
        }
        
        results_json = json.dumps(export_results, indent=2)
        st.download_button("📥 Download Evaluation Results (JSON)", results_json, "evaluation_results.json", "application/json")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.caption("Thesis Claim Analyzer v1.0 | Built for research paper analysis")