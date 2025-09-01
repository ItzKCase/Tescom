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
import asyncio
import time
from typing import Dict, Tuple, List, Any, Callable, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Thread-safe progress tracking for async operations."""
    
    def __init__(self, total_items: int):
        self.total_items = total_items
        self.completed_items = 0
        self.current_step = ""
        self.errors = []
        self.start_time = time.time()
        self.lock = asyncio.Lock()
        
    async def update(self, increment: int = 1, step: str = ""):
        async with self.lock:
            self.completed_items += increment
            if step:
                self.current_step = step
                
    async def add_error(self, error: str):
        async with self.lock:
            self.errors.append(error)
            
    async def get_status(self) -> Dict[str, Any]:
        async with self.lock:
            elapsed_time = time.time() - self.start_time
            progress_percent = (self.completed_items / self.total_items) * 100 if self.total_items > 0 else 0
            
            # Estimate remaining time based on current progress
            if self.completed_items > 0 and elapsed_time > 0:
                rate = self.completed_items / elapsed_time
                remaining_items = self.total_items - self.completed_items
                eta_seconds = remaining_items / rate if rate > 0 else 0
            else:
                eta_seconds = 0
                
            return {
                "total_items": self.total_items,
                "completed_items": self.completed_items,
                "progress_percent": round(progress_percent, 1),
                "current_step": self.current_step,
                "elapsed_time": round(elapsed_time, 1),
                "eta_seconds": round(eta_seconds, 1),
                "errors_count": len(self.errors),
                "is_complete": self.completed_items >= self.total_items
            }

class AsyncProgressCallback:
    """Callback interface for progress updates."""
    
    def __init__(self, callback_func: Optional[Callable] = None):
        self.callback_func = callback_func
        
    async def __call__(self, status: Dict[str, Any]):
        if self.callback_func:
            try:
                if asyncio.iscoroutinefunction(self.callback_func):
                    await self.callback_func(status)
                else:
                    self.callback_func(status)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

# Global progress tracking storage
_active_progress_trackers: Dict[str, ProgressTracker] = {}

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Intelligently detect manufacturer, model, and description columns using fuzzy matching.
    Returns mapping of detected columns.
    """
    column_mapping = {}
    
    # Get all available columns, including numeric ones
    all_columns = list(df.columns)
    logger.info(f"All columns in DataFrame: {all_columns}")
    logger.info(f"Column types: {[type(col) for col in all_columns]}")
    
    # First, try to find descriptive column names using pattern matching
    descriptive_columns = []
    for col in all_columns:
        col_str = str(col).strip()
        if pd.isna(col) or col_str == "":
            continue
        
        # Check if this looks like a descriptive column name
        col_lower = col_str.lower()
        if any(pattern in col_lower for pattern in ['manufacturer', 'mfr', 'mfg', 'brand', 'vendor', 'company', 'make', 'model', 'description', 'desc', 'type', 'category']):
            descriptive_columns.append(col)
    
    logger.info(f"Descriptive columns found: {descriptive_columns}")
    
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
    
    # If we found descriptive columns, use pattern matching
    if len(descriptive_columns) >= 3:
        logger.info("Using pattern matching for descriptive columns")
        
        # Find best matches for each required column type
        for col in descriptive_columns:
            col_lower = str(col).lower().strip()
            
            # Check manufacturer patterns
            if not column_mapping.get('manufacturer'):
                for pattern in manufacturer_patterns:
                    if pattern in col_lower or col_lower in pattern:
                        column_mapping['manufacturer'] = col
                        logger.info(f"Detected manufacturer column: {col}")
                        break
            
            # Check model patterns
            if not column_mapping.get('model'):
                for pattern in model_patterns:
                    if pattern in col_lower or col_lower in pattern:
                        column_mapping['model'] = col
                        logger.info(f"Detected model column: {col}")
                        break
            
            # Check description patterns
            if not column_mapping.get('description'):
                for pattern in description_patterns:
                    if pattern in col_lower or col_lower in pattern:
                        column_mapping['description'] = col
                        logger.info(f"Detected description column: {col}")
                        break
    
    logger.info(f"Final column mapping: {column_mapping}")
    
    # Fallback: if we couldn't detect columns by pattern, use the first three columns
    # This handles cases where the file has the right structure but different naming
    if len(column_mapping) < 3 and len(all_columns) >= 3:
        logger.info("Pattern matching incomplete, using fallback column mapping")
        if not column_mapping.get('manufacturer'):
            column_mapping['manufacturer'] = all_columns[0]
            logger.info(f"Fallback: using '{all_columns[0]}' as manufacturer")
        if not column_mapping.get('model'):
            column_mapping['model'] = all_columns[1]
            logger.info(f"Fallback: using '{all_columns[1]}' as model")
        if not column_mapping.get('description'):
            column_mapping['description'] = all_columns[2]
            logger.info(f"Fallback: using '{all_columns[2]}' as description")
    
    # CRITICAL: Ensure we have all required columns
    required_columns = ['manufacturer', 'model', 'description']
    missing_columns = [col for col in required_columns if col not in column_mapping]
    
    if missing_columns:
        logger.error(f"Missing required columns after detection: {missing_columns}")
        logger.error(f"Available all columns: {all_columns}")
        logger.error(f"Current column mapping: {column_mapping}")
        
        # Last resort: use any available columns, even if they're numeric
        if len(all_columns) >= 3:
            logger.warning("Using last resort column mapping with available columns")
            if 'manufacturer' not in column_mapping:
                column_mapping['manufacturer'] = all_columns[0]
            if 'model' not in column_mapping:
                column_mapping['model'] = all_columns[1]
            if 'description' not in column_mapping:
                column_mapping['description'] = all_columns[2]
        else:
            logger.error("Not enough columns available for processing")
            raise ValueError(f"Not enough columns available. Need at least 3, got {len(all_columns)}")
    
    logger.info(f"Final column mapping after fallback: {column_mapping}")
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
    
    if len(df) > 200:
        # Check for potential duplicates
        duplicate_check = check_for_duplicates(df, column_mapping)
        if duplicate_check:
            return False, f"File contains {len(df)} items, which exceeds the maximum limit of 200. {duplicate_check} Please consolidate your data and try again."
        else:
            return False, f"File contains {len(df)} items, which exceeds the maximum limit of 200. Please reduce the file size and try again."
    
    # Check for required data in each row
    required_cols = ['manufacturer', 'model', 'description']
    for col_type in required_cols:
        col_name = column_mapping[col_type]
        try:
            missing_data = df[col_name].isna().sum()
            if missing_data > 0:
                return False, f"Column '{col_name}' has {missing_data} missing values. All equipment must have manufacturer, model, and description information."
        except KeyError as e:
            return False, f"Column access error: {e}. This may indicate hidden or problematic columns in your Excel file. Please ensure your file has clean column headers."
        except Exception as e:
            return False, f"Error processing column '{col_name}': {e}. Please check your file format."
    
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
            try:
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
            except KeyError as e:
                logger.warning(f"Column access error in duplicate check: {e}")
                return "Unable to check for duplicates due to column access issues."
            except Exception as e:
                logger.warning(f"Error in duplicate check: {e}")
                return "Unable to check for duplicates due to processing error."
        
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
            # FINAL FIX: Multi-stage, intelligent header detection
            try:
                logger.info("=== FINAL FIX: Intelligent Header Detection ===")
                raw_df = pd.read_excel(file_path, engine='openpyxl', header=None, nrows=20)
                
                candidate_rows = []
                for i in range(min(10, len(raw_df))): # Scan first 10 rows
                    row_values = [str(v) for v in raw_df.iloc[i].dropna()]
                    if len(row_values) < 2: continue

                    score = 0
                    text_cells = 0
                    numeric_cells = 0
                    
                    for val in row_values:
                        val_lower = val.lower()
                        # Massive bonus for containing keywords
                        if any(k in val_lower for k in ['manufacturer', 'mfg', 'brand']): score += 30
                        if any(k in val_lower for k in ['model', 'part']): score += 30
                        if any(k in val_lower for k in ['description', 'desc']): score += 30
                        
                        # General text properties
                        if val.isalpha(): score += 5
                        if len(val) > 10: score += 5

                        # Penalize things that look like data, not headers
                        if val.replace('.', '', 1).isdigit():
                            numeric_cells += 1
                            score -= 15 # Penalize pure numbers
                        else:
                            text_cells += 1

                    # Structural score: headers have more text cells
                    if text_cells > numeric_cells:
                        score += 20
                    
                    logger.info(f"Row {i} scored {score} with values {row_values}")
                    candidate_rows.append({'row_index': i, 'score': score})

                # Sort by score and try the best candidates
                sorted_candidates = sorted(candidate_rows, key=lambda x: x['score'], reverse=True)
                
                df = None
                best_header_row = -1
                for candidate in sorted_candidates:
                    header_row_index = candidate['row_index']
                    logger.info(f"Attempting to use row {header_row_index} as header (score: {candidate['score']})")
                    temp_df = pd.read_excel(file_path, engine='openpyxl', header=header_row_index)
                    
                    # Sanity check the resulting column names
                    column_names = [str(c).strip() for c in temp_df.columns]
                    problematic_names = [c for c in column_names if c.isdigit() or len(c) < 2 or any(p in c for p in ['400PLUS', 'POSITECTOR'])]
                    
                    if not problematic_names:
                        logger.info(f"SUCCESS: Row {header_row_index} selected as header. Columns: {column_names}")
                        df = temp_df
                        best_header_row = header_row_index
                        break
                    else:
                        logger.warning(f"REJECTED: Row {header_row_index} resulted in problematic columns: {problematic_names}")

                if df is None:
                    logger.error("All header candidates were rejected. Falling back to row 0.")
                    df = pd.read_excel(file_path, engine='openpyxl', header=0)

            except Exception as e:
                logger.error(f"Failed intelligent header detection: {e}")
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                    logger.warning("Using fallback Excel reading.")
                except Exception as e2:
                    logger.error(f"All Excel reading methods failed: {e2}")
                    return "", f"❌ Failed to read Excel file: {str(e2)}"
        else:
            return "", "❌ Unsupported file format. Please upload a .csv or .xlsx file."
        
        # Log original columns for debugging
        logger.info(f"Original columns in file: {list(df.columns)}")
        logger.info(f"Column types: {[type(col) for col in df.columns]}")
        
        # Detect columns
        column_mapping = detect_columns(df)
        logger.info(f"Column mapping result: {column_mapping}")
        
        # CRITICAL FIX: Use the actual column names from the DataFrame
        # This handles Unicode character mismatches like 'Manufactureг' vs 'Manufacturer'
        actual_column_mapping = {}
        for col_type, detected_name in column_mapping.items():
            # Find the actual column name that matches the detected pattern
            for actual_col in df.columns:
                if actual_col == detected_name:
                    actual_column_mapping[col_type] = actual_col
                    break
            else:
                # If no exact match, use the first column that contains the detected name
                for actual_col in df.columns:
                    if detected_name.lower() in actual_col.lower() or actual_col.lower() in detected_name.lower():
                        actual_column_mapping[col_type] = actual_col
                        break
                else:
                    # Last resort: use the detected name as-is
                    actual_column_mapping[col_type] = detected_name
        
        logger.info(f"Original column mapping: {column_mapping}")
        logger.info(f"Actual column mapping using real column names: {actual_column_mapping}")
        
        # Use the actual column mapping for the rest of the process
        column_mapping = actual_column_mapping
        
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
                try:
                    cell_value = str(row[col]) if pd.notna(row[col]) else ""
                except KeyError as e:
                    logger.warning(f"Column access error in preview: {e}")
                    cell_value = "Error accessing column"
                preview_html += f"<td style='border: 1px solid #dee2e6; padding: 8px;'>{cell_value}</td>"
            preview_html += "</tr>"
        
        preview_html += "</table></div></div>"
        
        return preview_html, ""
        
    except Exception as e:
        logger.error(f"Error creating data preview: {e}")
        return "", f"❌ Error creating preview: {str(e)}"

def process_equipment_list_batch(df: pd.DataFrame, column_mapping: Dict[str, str], batch_size: int = 50) -> List[Dict[str, Any]]:
    """
    Process equipment list in batches for better memory management and performance.
    Returns list of processed equipment items.
    """
    from build_equipment_db import search_equipment
    
    results = []
    total_items = len(df)
    
    for batch_start in range(0, total_items, batch_size):
        batch_end = min(batch_start + batch_size, total_items)
        batch_df = df.iloc[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}: items {batch_start + 1}-{batch_end} of {total_items}")
        
        batch_results = []
        
        for idx, row in batch_df.iterrows():
            try:
                logger.info(f"Processing row {idx}, accessing columns: {column_mapping}")
                logger.info(f"Row index: {idx}, Row type: {type(row)}")
                logger.info(f"Row columns available: {list(row.index)}")
                logger.info(f"Row shape: {row.shape if hasattr(row, 'shape') else 'No shape'}")
                
                # Debug: Check what's in the row
                logger.info(f"Row content preview: {dict(row.head(3)) if hasattr(row, 'head') else dict(row)}")
                
                # Safe column access with validation
                if column_mapping['manufacturer'] not in row.index:
                    raise ValueError(f"Column '{column_mapping['manufacturer']}' not found in row. Available columns: {list(row.index)}")
                if column_mapping['model'] not in row.index:
                    raise ValueError(f"Column '{column_mapping['model']}' not found in row. Available columns: {list(row.index)}")
                if column_mapping['description'] not in row.index:
                    raise ValueError(f"Column '{column_mapping['description']}' not found in row. Available columns: {list(row.index)}")
                
                manufacturer = str(row[column_mapping['manufacturer']]).strip()
                model = str(row[column_mapping['model']]).strip()
                description = str(row[column_mapping['description']]).strip()
                
                # Search for equipment in database
                logger.info(f"About to search for: manufacturer='{manufacturer}', model='{model}'")
                try:
                    search_result = search_equipment(
                        db_path='equipment.db',
                        manufacturer_or_query=manufacturer,
                        model=model,
                        limit=1
                    )
                    logger.info(f"Search completed successfully for row {idx}")
                except Exception as search_error:
                    logger.error(f"Search error for row {idx}: {search_error}")
                    raise
                
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
                    
                    # Get database notes
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
                
                # Add to batch results
                batch_results.append({
                    'Original Manufacturer': manufacturer,
                    'Original Model': model,
                    'Original Description': description,
                    'Can Accredit': can_accredit,
                    'Match Type': match_type,
                    'Database Notes': db_notes,
                    'Agent Notes': agent_notes
                })
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                # Add error row to results with safe fallbacks
                try:
                    # Try to get row data safely with better error handling
                    manufacturer = 'Error'
                    model = 'Error'
                    description = 'Error'
                    
                    # Safely extract manufacturer
                    if 'manufacturer' in column_mapping and column_mapping['manufacturer'] in row.index:
                        try:
                            manufacturer = str(row[column_mapping['manufacturer']]).strip()
                        except:
                            manufacturer = 'Error'
                    
                    # Safely extract model
                    if 'model' in column_mapping and column_mapping['model'] in row.index:
                        try:
                            model = str(row[column_mapping['model']]).strip()
                        except:
                            model = 'Error'
                    
                    # Safely extract description
                    if 'description' in column_mapping and column_mapping['description'] in row.index:
                        try:
                            description = str(row[column_mapping['description']]).strip()
                        except:
                            description = 'Error'
                            
                except Exception as fallback_error:
                    logger.error(f"Error in fallback processing for row {idx}: {fallback_error}")
                    manufacturer = 'Error'
                    model = 'Error'
                    description = 'Error'
                
                batch_results.append({
                    'Original Manufacturer': manufacturer,
                    'Original Model': model,
                    'Original Description': description,
                    'Can Accredit': 'Error',
                    'Match Type': 'Processing Error',
                    'Database Notes': f'Error: {str(e)}',
                    'Agent Notes': 'Failed to process this row due to an error'
                })
        
        results.extend(batch_results)
        
        # Optional: force garbage collection after each batch
        import gc
        gc.collect()
    
    return results

async def process_equipment_item_async(item_data: Dict[str, str], progress_tracker: ProgressTracker) -> Dict[str, Any]:
    """
    Asynchronously process a single equipment item.
    """
    try:
        manufacturer = item_data['manufacturer']
        model = item_data['model'] 
        description = item_data['description']
        
        # Update progress
        await progress_tracker.update(step=f"Processing {manufacturer} {model}")
        
        # Run database search in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            from build_equipment_db import search_equipment
            search_result = await loop.run_in_executor(
                executor,
                search_equipment,
                'equipment.db',
                manufacturer,
                model,
                1
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
            
            # Get database notes
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
        
        # Update progress
        await progress_tracker.update(1)
        
        return {
            'Original Manufacturer': manufacturer,
            'Original Model': model,
            'Original Description': description,
            'Can Accredit': can_accredit,
            'Match Type': match_type,
            'Database Notes': db_notes,
            'Agent Notes': agent_notes
        }
        
    except Exception as e:
        error_msg = f"Error processing item {manufacturer} {model}: {str(e)}"
        await progress_tracker.add_error(error_msg)
        logger.error(error_msg)
        raise

async def process_equipment_list_async(
    file_path: str, 
    progress_callback: Optional[AsyncProgressCallback] = None,
    max_concurrent: int = 5
) -> Tuple[str, str, str]:
    """
    Asynchronously process equipment list with progress tracking.
    Returns (file_path, status_message, progress_id).
    """
    import uuid
    progress_id = str(uuid.uuid4())
    
    try:
        # Read and validate file
        if file_path.endswith('.csv'):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 5:
                # Use chunked reading for large files
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    df_chunks = await loop.run_in_executor(
                        executor, 
                        lambda: pd.read_csv(file_path, chunksize=1000)
                    )
                    df = await loop.run_in_executor(
                        executor,
                        lambda: pd.concat(df_chunks, ignore_index=True)
                    )
            else:
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    df = await loop.run_in_executor(executor, pd.read_csv, file_path)
        elif file_path.endswith('.xlsx'):
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                try:
                    # First try to read with header=0 (first row)
                    df = await loop.run_in_executor(
                        executor, 
                        lambda: pd.read_excel(file_path, engine='openpyxl', header=0)
                    )
                    
                    # Check if we got meaningful column names
                    if any(isinstance(col, (int, float)) for col in df.columns) or any(str(col).isdigit() for col in df.columns):
                        logger.info("Detected numeric column names, trying header=1 (second row)")
                        # Try reading with header=1 (second row) if first row gave numeric names
                        df = await loop.run_in_executor(
                            executor,
                            lambda: pd.read_excel(file_path, engine='openpyxl', header=1)
                        )
                        
                    # If still no good headers, try to find the header row automatically
                    if any(isinstance(col, (int, float)) for col in df.columns) or any(str(col).isdigit() for col in df.columns):
                        logger.info("Still numeric columns, trying to find header row automatically")
                        # Read first few rows to find the header
                        temp_df = await loop.run_in_executor(
                            executor,
                            lambda: pd.read_excel(file_path, engine='openpyxl', header=None, nrows=10)
                        )
                        
                        # Look for row with descriptive text (not empty, not numeric)
                        header_row = 0
                        for i in range(min(5, len(temp_df))):  # Check first 5 rows
                            row_values = temp_df.iloc[i].dropna()
                            if len(row_values) >= 3:  # Need at least 3 columns
                                # Check if this row has descriptive text
                                descriptive_count = sum(1 for val in row_values if isinstance(val, str) and len(str(val).strip()) > 2)
                                if descriptive_count >= 2:  # At least 2 descriptive columns
                                    header_row = i
                                    break
                        
                        logger.info(f"Using row {header_row} as header")
                        df = await loop.run_in_executor(
                            executor,
                            lambda: pd.read_excel(file_path, engine='openpyxl', header=header_row)
                        )
                        
                except Exception as e:
                    logger.warning(f"Failed to read with openpyxl: {e}, trying xlrd")
                    try:
                        df = await loop.run_in_executor(
                            executor,
                            lambda: pd.read_excel(file_path, engine='xlrd')
                        )
                    except Exception as e2:
                        logger.warning(f"Failed to read with xlrd: {e2}, trying default")
                        df = await loop.run_in_executor(
                            executor,
                            lambda: pd.read_excel(file_path)
                        )
        else:
            return "", "❌ Unsupported file format. Please upload a .csv or .xlsx file.", progress_id
        
        logger.info(f"Loaded {len(df)} equipment items for async processing")
        
        # Detect columns
        column_mapping = detect_columns(df)
        
        # CRITICAL FIX: Use the actual column names from the DataFrame
        # This handles Unicode character mismatches like 'Manufactureг' vs 'Manufacturer'
        actual_column_mapping = {}
        for col_type, detected_name in column_mapping.items():
            # Find the actual column name that matches the detected pattern
            for actual_col in df.columns:
                if actual_col == detected_name:
                    actual_column_mapping[col_type] = actual_col
                    break
            else:
                # If no exact match, use the first column that contains the detected name
                for actual_col in df.columns:
                    if detected_name.lower() in actual_col.lower() or actual_col.lower() in detected_name.lower():
                        actual_column_mapping[col_type] = actual_col
                        break
                else:
                    # Last resort: use the detected name as-is
                    actual_column_mapping[col_type] = detected_name
        
        logger.info(f"Original column mapping: {column_mapping}")
        logger.info(f"Actual column mapping using real column names: {actual_column_mapping}")
        
        # Use the actual column mapping for the rest of the process
        column_mapping = actual_column_mapping
        
        # Validate data
        is_valid, error_message = validate_equipment_data(df, column_mapping, file_path)
        if not is_valid:
            return "", f"❌ {error_message}", progress_id
        
        # Initialize progress tracker
        progress_tracker = ProgressTracker(len(df))
        _active_progress_trackers[progress_id] = progress_tracker
        
        await progress_tracker.update(step="Preparing equipment items for processing")
        
        # Prepare items for processing
        items = []
        for idx, row in df.iterrows():
            items.append({
                'manufacturer': str(row[column_mapping['manufacturer']]).strip(),
                'model': str(row[column_mapping['model']]).strip(),
                'description': str(row[column_mapping['description']]).strip()
            })
        
        await progress_tracker.update(step="Starting concurrent processing")
        
        # Process items concurrently with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await process_equipment_item_async(item, progress_tracker)
        
        # Create tasks for concurrent processing
        tasks = [process_with_semaphore(item) for item in items]
        
        # Process with progress updates
        results = []
        completed = 0
        
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                completed += 1
                
                # Send progress update via callback
                if progress_callback:
                    status = await progress_tracker.get_status()
                    await progress_callback(status)
                    
                # Log progress every 10 items
                if completed % 10 == 0:
                    logger.info(f"Completed {completed}/{len(items)} equipment items")
                    
            except Exception as e:
                logger.error(f"Failed to process equipment item: {e}")
                await progress_tracker.add_error(str(e))
        
        await progress_tracker.update(step="Creating results file")
        
        # Create results DataFrame with memory-efficient operations
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            results_df = await loop.run_in_executor(executor, pd.DataFrame, results)
        
        # Clear original dataframe from memory
        del df
        import gc
        gc.collect()
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            with ThreadPoolExecutor() as executor:
                await loop.run_in_executor(
                    executor,
                    lambda: results_df.to_excel(
                        tmp_file.name, 
                        index=False, 
                        engine='openpyxl'
                    )
                )
            
            logger.info(f"Created temporary result file: {tmp_file.name}")
            
            await progress_tracker.update(step="Processing complete")
            
            # Final progress update
            if progress_callback:
                status = await progress_tracker.get_status()
                await progress_callback(status)
            
            return tmp_file.name, f"✅ Successfully processed {len(results)} equipment items. Results ready for download.", progress_id
    
    except Exception as e:
        error_msg = f"❌ Error processing file: {str(e)}"
        logger.error(error_msg)
        if progress_id in _active_progress_trackers:
            await _active_progress_trackers[progress_id].add_error(error_msg)
        return "", error_msg, progress_id

def get_progress_status(progress_id: str) -> Optional[Dict[str, Any]]:
    """
    Get current progress status for a processing job.
    """
    if progress_id in _active_progress_trackers:
        # Run async function in event loop
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_active_progress_trackers[progress_id].get_status())
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(_active_progress_trackers[progress_id].get_status())
    return None

def cleanup_progress_tracker(progress_id: str):
    """
    Clean up completed progress tracker.
    """
    if progress_id in _active_progress_trackers:
        del _active_progress_trackers[progress_id]

def process_equipment_list(file_path: str) -> Tuple[str, str]:
    """
    Process uploaded equipment list with enhanced batch processing and memory management.
    Returns (file_path, status_message).
    """
    try:
        # Determine file type and read with chunking for large files
        if file_path.endswith('.csv'):
            # Check file size and use chunking if needed
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 5:  # Use chunking for files larger than 5MB
                df_chunks = pd.read_csv(file_path, chunksize=1000)
                df = pd.concat(df_chunks, ignore_index=True)
            else:
                df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            # FINAL FIX: Multi-stage, intelligent header detection
            try:
                logger.info("=== FINAL FIX: Intelligent Header Detection ===")
                raw_df = pd.read_excel(file_path, engine='openpyxl', header=None, nrows=20)
                
                candidate_rows = []
                for i in range(min(10, len(raw_df))): # Scan first 10 rows
                    row_values = [str(v) for v in raw_df.iloc[i].dropna()]
                    if len(row_values) < 2: continue

                    score = 0
                    text_cells = 0
                    numeric_cells = 0
                    
                    for val in row_values:
                        val_lower = val.lower()
                        # Massive bonus for containing keywords
                        if any(k in val_lower for k in ['manufacturer', 'mfg', 'brand']): score += 30
                        if any(k in val_lower for k in ['model', 'part']): score += 30
                        if any(k in val_lower for k in ['description', 'desc']): score += 30
                        
                        # General text properties
                        if val.isalpha(): score += 5
                        if len(val) > 10: score += 5

                        # Penalize things that look like data, not headers
                        if val.replace('.', '', 1).isdigit():
                            numeric_cells += 1
                            score -= 15 # Penalize pure numbers
                        else:
                            text_cells += 1

                    # Structural score: headers have more text cells
                    if text_cells > numeric_cells:
                        score += 20
                    
                    logger.info(f"Row {i} scored {score} with values {row_values}")
                    candidate_rows.append({'row_index': i, 'score': score})

                # Sort by score and try the best candidates
                sorted_candidates = sorted(candidate_rows, key=lambda x: x['score'], reverse=True)
                
                df = None
                best_header_row = -1
                for candidate in sorted_candidates:
                    header_row_index = candidate['row_index']
                    logger.info(f"Attempting to use row {header_row_index} as header (score: {candidate['score']})")
                    temp_df = pd.read_excel(file_path, engine='openpyxl', header=header_row_index)
                    
                    # Sanity check the resulting column names
                    column_names = [str(c).strip() for c in temp_df.columns]
                    problematic_names = [c for c in column_names if c.isdigit() or len(c) < 2 or any(p in c for p in ['400PLUS', 'POSITECTOR'])]
                    
                    if not problematic_names:
                        logger.info(f"SUCCESS: Row {header_row_index} selected as header. Columns: {column_names}")
                        df = temp_df
                        best_header_row = header_row_index
                        break
                    else:
                        logger.warning(f"REJECTED: Row {header_row_index} resulted in problematic columns: {problematic_names}")

                if df is None:
                    logger.error("All header candidates were rejected. Falling back to row 0.")
                    df = pd.read_excel(file_path, engine='openpyxl', header=0)

            except Exception as e:
                logger.error(f"Failed intelligent header detection: {e}")
                try:
                    df = pd.read_excel(file_path, engine='openpyxl')
                    logger.warning("Using fallback Excel reading.")
                except Exception as e2:
                    logger.error(f"All Excel reading methods failed: {e2}")
                    return "", f"❌ Failed to read Excel file: {str(e2)}"
        else:
            return "", "❌ Unsupported file format. Please upload a .csv or .xlsx file."
        
        # Log original columns for debugging
        logger.info(f"Original columns in file: {list(df.columns)}")
        logger.info(f"Column types: {[type(col) for col in df.columns]}")
        logger.info(f"Processing {len(df)} equipment items")
        
        # Detect columns
        column_mapping = detect_columns(df)
        logger.info(f"Column mapping result: {column_mapping}")
        
        # CRITICAL FIX: Use the actual column names from the DataFrame
        # This handles Unicode character mismatches like 'Manufactureг' vs 'Manufacturer'
        actual_column_mapping = {}
        for col_type, detected_name in column_mapping.items():
            # Find the actual column name that matches the detected pattern
            for actual_col in df.columns:
                if actual_col == detected_name:
                    actual_column_mapping[col_type] = actual_col
                    break
            else:
                # If no exact match, use the first column that contains the detected name
                for actual_col in df.columns:
                    if detected_name.lower() in actual_col.lower() or actual_col.lower() in detected_name.lower():
                        actual_column_mapping[col_type] = actual_col
                        break
                else:
                    # Last resort: use the detected name as-is
                    actual_column_mapping[col_type] = detected_name
        
        logger.info(f"Original column mapping: {column_mapping}")
        logger.info(f"Actual column mapping using real column names: {actual_column_mapping}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        
        # Verify column mapping is valid
        for col_type, col_name in actual_column_mapping.items():
            if col_name not in df.columns:
                error_msg = f"Column mapping error: '{col_type}' maps to '{col_name}' but this column doesn't exist in DataFrame. Available columns: {list(df.columns)}"
                logger.error(error_msg)
                return "", f"❌ {error_msg}"
        
        # Additional validation: check if column names are problematic
        problematic_columns = []
        for col_type, col_name in actual_column_mapping.items():
            if isinstance(col_name, (int, float)) or str(col_name).isdigit():
                problematic_columns.append(f"{col_type}: {col_name}")
        
        if problematic_columns:
            logger.warning(f"Detected potentially problematic column names: {problematic_columns}")
            logger.warning("These numeric column names may cause processing issues")
        
        # Use the actual column mapping for the rest of the process
        column_mapping = actual_column_mapping
        
        is_valid, error_message = validate_equipment_data(df, column_mapping, file_path)
        if not is_valid:
            return "", f"❌ {error_message}"
        
        # Process equipment in batches for better performance
        batch_size = 25 if len(df) > 100 else 50  # Smaller batches for larger datasets
        logger.info(f"Starting batch processing with column mapping: {column_mapping}")
        logger.info(f"DataFrame columns: {list(df.columns)}")
        logger.info(f"DataFrame shape: {df.shape}")
        
        # CRITICAL DEBUG: Check for any hidden or problematic columns
        logger.info("=== CRITICAL DEBUG: DataFrame Analysis ===")
        for i, col in enumerate(df.columns):
            logger.info(f"Column {i}: '{col}' (type: {type(col)}, repr: {repr(col)})")
            if isinstance(col, (int, float)) or str(col).isdigit():
                logger.error(f"PROBLEMATIC COLUMN DETECTED: {col} at index {i}")
        
        # Check if column mapping columns actually exist
        for col_type, col_name in column_mapping.items():
            if col_name not in df.columns:
                logger.error(f"COLUMN MAPPING ERROR: '{col_type}' maps to '{col_name}' but this column doesn't exist!")
                logger.error(f"Available columns: {list(df.columns)}")
                return "", f"❌ Column mapping error: '{col_type}' maps to '{col_name}' but this column doesn't exist in the DataFrame."
        
        # Additional debug: Show the final column mapping that will be used
        logger.info("=== FINAL COLUMN MAPPING FOR PROCESSING ===")
        for col_type, col_name in column_mapping.items():
            logger.info(f"  {col_type}: '{col_name}' (exists in DataFrame: {col_name in df.columns})")
        
        results = process_equipment_list_batch(df, column_mapping, batch_size)
        
        # Create results DataFrame with memory-efficient operations
        results_df = pd.DataFrame(results)
        
        # Clear the original dataframe from memory
        del df
        import gc
        gc.collect()
        
        # Create temporary file for download
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            # Use efficient Excel writing with minimal memory usage
            with pd.ExcelWriter(tmp_file.name, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Equipment Assessment')
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
