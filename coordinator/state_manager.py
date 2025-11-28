import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
from utils.file_utils import write_json_file, append_to_json_log, get_iso_timestamp


class StateManager:
    """
    Manages workflow state using local JSON files (Redis replacement).
    Handles state persistence, audit logging, and expiration tracking.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize state manager.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_discharge_state(
        self,
        patient_id: str,
        status: str,
        agents_output: Dict[str, Any],
        issues: List[Dict[str, Any]],
        final_decision: str,
        approved_by: List[str],
        blocked_by: List[str]
    ) -> str:
        """
        Save discharge workflow state to file.
        
        Args:
            patient_id: Patient identifier
            status: Status (approved|hold|pending_auto_resolution)
            agents_output: Raw agent outputs
            issues: Aggregated issues
            final_decision: Final decision
            approved_by: List of agents that approved
            blocked_by: List of agents that blocked
            
        Returns:
            Path to the saved state file
        """
        timestamp = get_iso_timestamp()
        
        # Calculate expiration (6 hours from now if approved)
        expires_at = None
        if status == "approved":
            expiry_time = datetime.now() + timedelta(hours=6)
            expires_at = expiry_time.isoformat()
        
        state = {
            "patient_id": patient_id,
            "status": status,
            "final_decision": final_decision,
            "timestamp": timestamp,
            "expires_at": expires_at,
            "approved_by": approved_by,
            "blocked_by": blocked_by,
            "agents": agents_output,
            "issues": issues,
            "metadata": {
                "created_at": timestamp,
                "last_updated": timestamp,
                "version": "1.0"
            }
        }
        
        file_path = self.output_dir / f"discharge_state_{patient_id}.json"
        write_json_file(str(file_path), state)
        
        return str(file_path)
    
    def append_audit_log(
        self,
        patient_id: str,
        final_decision: str,
        issues: List[Dict[str, Any]],
        recommended_next_steps: List[str]
    ) -> str:
        """
        Append entry to audit log.
        
        Args:
            patient_id: Patient identifier
            final_decision: Final decision made
            issues: List of issues
            recommended_next_steps: Suggested actions
            
        Returns:
            Path to the audit log file
        """
        timestamp = get_iso_timestamp()
        
        entry = {
            "timestamp": timestamp,
            "patient_id": patient_id,
            "final_decision": final_decision,
            "issues_count": len(issues),
            "critical_issues": [i for i in issues if i.get("severity") == "critical"],
            "recommended_next_steps": recommended_next_steps
        }
        
        file_path = self.output_dir / f"discharge_audit_log_{patient_id}.json"
        append_to_json_log(str(file_path), entry)
        
        return str(file_path)
    
    def load_discharge_state(self, patient_id: str) -> Dict[str, Any]:
        """
        Load discharge state from file.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            State dictionary or None if not found
        """
        file_path = self.output_dir / f"discharge_state_{patient_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading state: {e}")
            return None
    
    def is_state_expired(self, patient_id: str) -> bool:
        """
        Check if discharge approval has expired.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            True if expired, False otherwise
        """
        state = self.load_discharge_state(patient_id)
        
        if not state or not state.get("expires_at"):
            return False
        
        try:
            expires_at = datetime.fromisoformat(state["expires_at"])
            return datetime.now() > expires_at
        except Exception:
            return False
