from typing import Dict, Any, List, TypedDict
from typing_extensions import Annotated
import operator


class DischargeState(TypedDict):
    """
    State schema for the discharge workflow.
    Used by LangGraph to track workflow progress.
    """
    # Patient information
    patient_id: str
    
    # Agent outputs (each agent writes to its own key)
    insurance_output: Dict[str, Any]
    pharmacy_output: Dict[str, Any]
    ambulance_output: Dict[str, Any]
    bed_output: Dict[str, Any]
    lab_output: Dict[str, Any]
    
    # Aggregated results
    all_agents_complete: bool
    aggregated_issues: Annotated[List[Dict[str, Any]], operator.add]
    
    # Final decision
    final_decision: str  # "APPROVE" | "HOLD" | "PENDING_AUTO_RESOLUTION"
    approved: bool
    approved_by: List[str]
    blocked_by: List[str]
    
    # Discharge summary
    discharge_summary: Dict[str, str]
    
    # Auto-resolution suggestions
    suggested_auto_resolutions: List[Dict[str, Any]]
    
    # Metadata
    timestamp: str
    files_written: List[str]


def create_initial_state(patient_id: str) -> DischargeState:
    """
    Create initial workflow state.
    
    Args:
        patient_id: Patient identifier
        
    Returns:
        Initial DischargeState
    """
    return DischargeState(
        patient_id=patient_id,
        insurance_output={},
        pharmacy_output={},
        ambulance_output={},
        bed_output={},
        lab_output={},
        all_agents_complete=False,
        aggregated_issues=[],
        final_decision="",
        approved=False,
        approved_by=[],
        blocked_by=[],
        discharge_summary={},
        suggested_auto_resolutions=[],
        timestamp="",
        files_written=[]
    )
