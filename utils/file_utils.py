import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


def read_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Safely read a JSON file and return its contents.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data, or None if file doesn't exist or is invalid
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {file_path}: {e}")
        return None


def write_json_file(file_path: str, data: Dict[str, Any], indent: int = 2) -> bool:
    """
    Safely write data to a JSON file.
    
    Args:
        file_path: Path to the JSON file
        data: Dictionary to write
        indent: JSON indentation level
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"Error writing {file_path}: {e}")
        return False


def append_to_json_log(file_path: str, entry: Dict[str, Any]) -> bool:
    """
    Append an entry to a JSON log file (array of entries).
    
    Args:
        file_path: Path to the log file
        entry: Dictionary entry to append
        
    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(file_path)
        
        # Read existing log or create new array
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                log = json.load(f)
        else:
            log = []
        
        # Append new entry
        log.append(entry)
        
        # Write back
        return write_json_file(file_path, log)
    except Exception as e:
        print(f"Error appending to {file_path}: {e}")
        return False


def format_evidence_path(file_path: str, json_path: Optional[str] = None, line_range: Optional[tuple] = None) -> str:
    """
    Format an evidence reference path.
    
    Args:
        file_path: Base file path
        json_path: Optional JSON path (e.g., "results[0].status")
        line_range: Optional tuple of (start_line, end_line)
        
    Returns:
        Formatted evidence string
    """
    if json_path:
        return f"{file_path}#{json_path}"
    elif line_range:
        start, end = line_range
        return f"{file_path}#line:{start}-{end}"
    else:
        return file_path


def get_iso_timestamp() -> str:
    """
    Get current timestamp in ISO8601 format.
    
    Returns:
        ISO8601 formatted timestamp string
    """
    return datetime.now().isoformat()


def calculate_elapsed_ms(start_time: datetime) -> float:
    """
    Calculate elapsed time in milliseconds.
    
    Args:
        start_time: Starting datetime
        
    Returns:
        Elapsed time in milliseconds
    """
    elapsed = datetime.now() - start_time
    return elapsed.total_seconds() * 1000
