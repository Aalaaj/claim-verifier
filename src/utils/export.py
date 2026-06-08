"""Excel export utilities"""

import pandas as pd
from typing import List, Dict
import numpy as np


class ExcelExporter:
    """Export analysis results to Excel"""
    
    def export(self, analysis, filename: str):
        """Export PaperAnalysis to Excel file"""
        
        # Convert claims to DataFrame
        claims_dicts = []
        for claim in analysis.claims:
            claim_dict = claim.to_dict()
            if 'entities' in claim_dict and claim_dict['entities']:
                claim_dict['entities'] = str(claim_dict['entities'])
            claims_dicts.append(claim_dict)
        
        df = pd.DataFrame(claims_dicts)
        
        # Clean data types for Excel
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(self._safe_string_conversion)
        
        # Create summary dataframe
        summary_data = []
        for key, value in analysis.overall_statistics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    summary_data.append({'Metric': f"{key}_{subkey}", 'Value': subvalue})
            else:
                summary_data.append({'Metric': key, 'Value': value})
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to Excel
        try:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='All Claims', index=False)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add filtered sheets
                if 'verification_status' in df.columns:
                    verified = df[df['verification_status'] == 'Verified']
                    if not verified.empty:
                        verified.to_excel(writer, sheet_name='Verified Claims', index=False)
                
                if 'importance' in df.columns:
                    critical = df[df['importance'] == 'Critical']
                    if not critical.empty:
                        critical.to_excel(writer, sheet_name='Critical Claims', index=False)
            
            print(f"\n✅ Results exported to: {filename}")
        except Exception as e:
            print(f"\n❌ Error exporting to Excel: {e}")
            csv_filename = filename.replace('.xlsx', '.csv')
            df.to_csv(csv_filename, index=False)
            print(f"✅ Saved as CSV instead: {csv_filename}")
    
    def _safe_string_conversion(self, x):
        """Safely convert any value to string"""
        # Handle None
        if x is None:
            return ""
        
        # Handle pandas NA - wrap in try/except
        try:
            if pd.isna(x):
                return ""
        except (ValueError, TypeError):
            pass
        
        # Handle empty numpy arrays (this is your specific error)
        if isinstance(x, np.ndarray):
            if x.size == 0:
                return ""
            return str(x.tolist())[:500]
        
        # Handle lists and tuples
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return ""
            return str(x)[:500]
        
        # Handle dicts
        if isinstance(x, dict):
            if len(x) == 0:
                return ""
            return str(x)[:500]
        
        # Everything else
        try:
            return str(x)[:500]
        except:
            return ""