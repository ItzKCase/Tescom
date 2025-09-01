"""
Equipment List Processing Module

This module handles the processing of uploaded equipment lists (Excel/CSV) and
generates accreditation assessment reports by matching equipment against the
Tescom equipment database.
"""

import pandas as pd
import tempfile
import logging
import os
from typing import Dict, Tuple, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Intelligently detect manufacturer, model, and description columns using fuzzy matching.
    Returns mapping of detected columns.
    """
    column_mapping = {}
    columns = [col.lower().strip() for col in df.columns]
    
    # Manufacturer column detection patterns
    manufacturer_patterns = [
        'manufacturer', 'mfr', 'mfg', 'brand', 'vendor', 'company', 'make',
        'manufacturer name', 'mfr name', 'brand name'
    ]
    
    # Model column detection patterns  
    model_patterns = [
        'model', 'model#', 'model #', 'model number', 'part number', 'pn', 'p/n',
        'serial', 'serial number', 's/n', 's/number', 'item', 'item number'
    ]
    
    # Description column detection patterns
    description_patterns = [
        'description', 'desc', 'type', 'category', 'equipment', 'instrument',
        'device', 'tool', 'name', 'title', 'specification', 'spec'
    ]
    
    # Find best matches for each required column type
    for col in df.columns:
        col_lower = col.lower().strip()
        
        # Check manufacturer patterns
        if not column_mapping.get('manufacturer'):
            for pattern in manufacturer_patterns:
                if pattern in col_lower or col_lower in pattern:
                    column_mapping['manufacturer'] = col
                    break
        
        # Check model patterns
        if not column_mapping.get('model'):
            for pattern in model_patterns:
                if pattern in col_lower or col_lower in pattern:
                    column_mapping['model'] = col
                    break
        
        # Check description patterns
        if not column_mapping.get('description'):
            for pattern in description_patterns:
                if pattern in col_lower or col_lower in pattern:
                    column_mapping['description'] = col
                    break
    
    return column_mapping

def validate_equipment_data(df: pd.DataFrame, column_mapping: Dict[str, str], file_path: str = None) -> Tuple[bool, str]:
    """
    Validate that the equipment data has required information.
    Returns (is_valid, error_message).
    """
    # Check file size (10MB limit)
    if file_path and os.path.exists(file_path):
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 10:
            return False, f"File size ({file_size_mb:.1f}MB) exceeds the 10MB limit. Please compress your file or reduce the number of items."
    
    if not column_mapping.get('manufacturer'):
        return False, "Could not identify manufacturer column. Please ensure your file has a column for manufacturer information."
    
    if not column_mapping.get('model'):
        return False, "Could not identify model column. Please ensure your file has a column for model information."
    
    if not column_mapping.get('description'):
        return False, "Could not identify description column. Please ensure your file has a column for equipment description."
    
    # Check if we have data
    if len(df) == 0:
        return False, "The uploaded file contains no data rows."
    
    if len(df) > 100:
        # Check for potential duplicates
        duplicate_check = check_for_duplicates(df, column_mapping)
        if duplicate_check:
            return False, f"File contains {len(df)} items, which exceeds the maximum limit of 100. {duplicate_check} Please consolidate your data and try again."
        else:
            return False, f"File contains {len(df)} items, which exceeds the maximum limit of 100. Please reduce the file size and try again."
    
    # Check for required data in each row
    required_cols = ['manufacturer', 'model', 'description']
    for col_type in required_cols:
        col_name = column_mapping[col_type]
        missing_data = df[col_name].isna().sum()
        if missing_data > 0:
            return False, f"Column '{col_name}' has {missing_data} missing values. All equipment must have manufacturer, model, and description information."
    
    return True, ""

def check_for_duplicates(df: pd.DataFrame, column_mapping: Dict[str, str]) -> str:
    """
    Check for potential duplicate entries in the equipment list.
    Returns a message if duplicates are found, empty string otherwise.
    """
    try:
        # Create a combined key for duplicate detection
        mfr_col = column_mapping.get('manufacturer', '')
        model_col = column_mapping.get('model', '')
        desc_col = column_mapping.get('description', '')
        
        if mfr_col and model_col:
            # Check for exact duplicates
            combined_key = df[mfr_col].astype(str) + '|' + df[model_col].astype(str)
            duplicates = combined_key.duplicated().sum()
            
            if duplicates > 0:
                return f"Found {duplicates} potential duplicate entries (same manufacturer and model)."
            
            # Check for similar entries (case-insensitive)
            mfr_lower = df[mfr_col].astype(str).str.lower()
            model_lower = df[model_col].astype(str).str.lower()
            combined_lower = mfr_lower + '|' + model_lower
            similar_duplicates = combined_lower.duplicated().sum()
            
            if similar_duplicates > 0:
                return f"Found {similar_duplicates} similar entries that might be duplicates (check case sensitivity)."
        
        return ""
    except Exception as e:
        logger.warning(f"Error checking for duplicates: {e}")
        return ""

def get_data_preview(file_path: str) -> Tuple[str, str]:
    """
    Get a preview of the uploaded data without processing.
    Returns (preview_html, error_message).
    """
    try:
        # Determine file type and read
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return "", "❌ Unsupported file format. Please upload a .csv or .xlsx file."
        
        # Detect columns
        column_mapping = detect_columns(df)
        
        # Create preview HTML
        preview_rows = min(5, len(df))  # Show first 5 rows
        preview_df = df.head(preview_rows)
        
        # Format the preview
        preview_html = f"""
        <div style="display: block;">
        <h4>📋 Data Preview ({len(df)} total rows)</h4>
        <p><strong>Detected Columns:</strong></p>
        <ul>
        """
        
        for col_type, col_name in column_mapping.items():
            preview_html += f"<li><strong>{col_type.title()}:</strong> {col_name}</li>"
        
        preview_html += "</ul>"
        
        if len(df) > preview_rows:
            preview_html += f"<p><em>Showing first {preview_rows} rows of {len(df)} total rows</em></p>"
        
        # Add table preview
        preview_html += "<div style='overflow-x: auto;'><table style='border-collapse: collapse; width: 100%;'>"
        
        # Header row
        preview_html += "<tr style='background-color: #f8f9fa;'>"
        for col in preview_df.columns:
            preview_html += f"<th style='border: 1px solid #dee2e6; padding: 8px; text-align: left;'>{col}</th>"
        preview_html += "</tr>"
        
        # Data rows
        for _, row in preview_df.iterrows():
            preview_html += "<tr>"
            for col in preview_df.columns:
                cell_value = str(row[col]) if pd.notna(row[col]) else ""
                preview_html += f"<td style='border: 1px solid #dee2e6; padding: 8px;'>{cell_value}</td>"
            preview_html += "</tr>"
        
        preview_html += "</table></div></div>"
        
        return preview_html, ""
        
    except Exception as e:
        logger.error(f"Error creating data preview: {e}")
        return "", f"❌ Error creating preview: {str(e)}"

def process_equipment_list(file_path: str) -> Tuple[str, str]:
    """
    Process uploaded equipment list and return results file path and status message.
    Returns (file_path, status_message).
    """
    try:
        # Determine file type and read
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return "", "❌ Unsupported file format. Please upload a .csv or .xlsx file."
        
        # Detect columns
        column_mapping = detect_columns(df)
        
        # Validate data
        is_valid, error_message = validate_equipment_data(df, column_mapping, file_path)
        if not is_valid:
            return "", f"❌ {error_message}"
        
        # Import search function
        from build_equipment_db import search_equipment
        
        # Process each equipment item
        results = []
        for idx, row in df.iterrows():
            manufacturer = str(row[column_mapping['manufacturer']]).strip()
            model = str(row[column_mapping['model']]).strip()
            description = str(row[column_mapping['description']]).strip()
            
            # Search for equipment in database
            search_result = search_equipment(
                db_path='equipment.db',
                manufacturer_or_query=manufacturer,
                model=model,
                limit=1
            )
            
            # Determine match confidence and type
            if search_result['matches']:
                match = search_result['matches'][0]
                confidence = match.get('confidence', 0.0)
                
                if confidence >= 0.8:
                    match_type = "Exact Match"
                    can_accredit = "Yes" if match.get('accredited') else "No"
                elif confidence >= 0.6:
                    match_type = "Vendor Family Match"
                    can_accredit = "Yes" if match.get('accredited') else "No"
                elif confidence >= 0.3:
                    match_type = "Fuzzy Match"
                    can_accredit = "Yes" if match.get('accredited') else "No"
                else:
                    match_type = "Low Confidence"
                    can_accredit = "Uncertain"
                
                # Get database notes (you may need to extend this based on your schema)
                db_notes = ""
                if match.get('accredited') == 0:
                    db_notes = "Not accredited"
                elif match.get('accredited') == 1:
                    db_notes = "Accredited"
                
                # Generate agent notes
                if confidence >= 0.8:
                    agent_notes = f"High confidence match found. {can_accredit} for accreditation."
                elif confidence >= 0.6:
                    agent_notes = f"Vendor family match found. {can_accredit} for accreditation. Recommend verification."
                elif confidence >= 0.3:
                    agent_notes = f"Fuzzy match found. {can_accredit} for accreditation. Manual verification required."
                else:
                    agent_notes = "Low confidence match. Manual verification required."
                
            else:
                match_type = "No Match Found"
                can_accredit = "Unknown"
                db_notes = ""
                agent_notes = "No matching equipment found in database. May need custom assessment."
            
            # Add to results
            results.append({
                'Original Manufacturer': manufacturer,
                'Original Model': model,
                'Original Description': description,
                'Can Accredit': can_accredit,
                'Match Type': match_type,
                'Database Notes': db_notes,
                'Agent Notes': agent_notes
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            results_df.to_excel(tmp_file.name, index=False, engine='openpyxl')
            logger.info(f"Created temporary result file: {tmp_file.name}")
            return tmp_file.name, f"✅ Successfully processed {len(results)} equipment items. Results ready for download."
    
    except Exception as e:
        logger.error(f"Error processing equipment list: {e}")
        return "", f"❌ Error processing file: {str(e)}"

def cleanup_temp_files(temp_files: List[str]):
    """
    Clean up temporary files created during processing.
    """
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {temp_file}: {e}")

# Global list to track temporary files for cleanup
_temp_files = []

def add_temp_file(temp_file: str):
    """Add a temporary file to the cleanup list."""
    global _temp_files
    _temp_files.append(temp_file)

def cleanup_all_temp_files():
    """Clean up all tracked temporary files."""
    global _temp_files
    cleanup_temp_files(_temp_files)
    _temp_files = []
