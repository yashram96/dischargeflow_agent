from abc import ABC, abstractmethod
from typing import Dict, Any, List
from datetime import datetime
import json

from schemas.agent_schema import AgentOutputSchema, IssueSchema, AgentMetadata
from utils.file_utils import read_json_file, format_evidence_path, calculate_elapsed_ms


class BaseAgent(ABC):
    """
    Base class for all discharge verification agents.
    Provides common utilities and enforces standard output schema.
    """
    
    def __init__(self, agent_name: str):
        """
        Initialize the base agent.
        
        Args:
            agent_name: Name of the agent (e.g., "Insurance", "Pharmacy")
        """
        self.agent_name = agent_name
        self.start_time = None
        self.checked_fields = []
        
    def start_timer(self):
        """Start the execution timer"""
        self.start_time = datetime.now()
        
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.start_time is None:
            return 0.0
        return calculate_elapsed_ms(self.start_time)
    
    def add_checked_field(self, field_name: str):
        """Add a field to the list of checked fields"""
        if field_name not in self.checked_fields:
            self.checked_fields.append(field_name)
    
    def create_issue(
        self,
        code: str,
        title: str,
        severity: str,
        message: str,
        suggested_action: str,
        evidence: List[str] = None,
        data: Dict[str, Any] = None
    ) -> IssueSchema:
        """
        Create a standardized issue object.
        
        Args:
            code: Unique issue code
            title: Short title
            severity: "low", "medium", "high", or "critical"
            message: Human-readable explanation
            suggested_action: What should be done
            evidence: List of evidence paths
            data: Supporting data
            
        Returns:
            IssueSchema object
        """
        return IssueSchema(
            code=code,
            title=title,
            severity=severity,
            message=message,
            suggested_action=suggested_action,
            evidence=evidence or [],
            data=data or {}
        )
    
    def create_output(
        self,
        noc: bool,
        confidence: float,
        issues: List[IssueSchema] = None,
        raw_response: Dict[str, Any] = None
    ) -> AgentOutputSchema:
        """
        Create standardized agent output.
        
        Args:
            noc: Whether NOC is granted
            confidence: Confidence score (0.0 to 1.0)
            issues: List of issues found
            raw_response: Optional raw data
            
        Returns:
            AgentOutputSchema object
        """
        meta = AgentMetadata(
            checked_fields=self.checked_fields,
            time_ms=self.get_elapsed_ms(),
            raw_response=raw_response or {}
        )
        
        return AgentOutputSchema(
            agent=self.agent_name,
            noc=noc,
            confidence=confidence,
            issues=issues or [],
            meta=meta
        )
    
    def load_patient_data(self, patient_id: str = None) -> Dict[str, Any]:
        """
        Load patient data from JSON file.
        
        Args:
            patient_id: Optional patient ID to filter for. If None, uses self.patient_id.
            
        Returns:
            Dict containing patient data
        """
        target_id = patient_id or getattr(self, 'patient_id', None)
        
        try:
            data = read_json_file("patient_data.json")
            
            # Handle list of patients
            if isinstance(data, list):
                if not target_id:
                    # If no ID specified and data is list, return first one or raise error
                    # For safety, let's default to P00231 if we can't find one
                    return data[0]
                
                # Find specific patient
                for patient in data:
                    # Check various common locations for ID
                    pid = (patient.get("Patient Information", {}).get("Patient ID") or 
                           patient.get("patient_id") or 
                           patient.get("id"))
                    
                    if pid == target_id:
                        return patient
                
                print(f"⚠️  Patient ID {target_id} not found in patient_data.json")
                return {}
            
            return data
            
        except Exception as e:
            print(f"Error loading patient data: {e}")
            return {}

    def get_patient_record(self, data: List[Dict], patient_id: str, id_field: str = "patient_id") -> Dict[str, Any]:
        """
        Helper to find a specific patient's record in a list of records.
        
        Args:
            data: List of dictionaries
            patient_id: ID to look for
            id_field: Field name containing the ID (default: "patient_id")
            
        Returns:
            Matching record or empty dict
        """
        if not isinstance(data, list):
            return data if data else {}
            
        for record in data:
            if record.get(id_field) == patient_id:
                return record
        return {}
    
    @abstractmethod
    def verify(self, patient_id: str, **kwargs) -> AgentOutputSchema:
        """
        Main verification method - must be implemented by each agent.
        
        Args:
            patient_id: Patient identifier
            **kwargs: Additional agent-specific parameters
            
        Returns:
            AgentOutputSchema with verification results
        """
        pass

    def to_json(self, output: AgentOutputSchema) -> str:
        """Convert output to JSON string"""
        return output.model_dump_json(indent=2)
    
    def to_dict(self, output: AgentOutputSchema) -> Dict[str, Any]:
        """Convert output to dictionary"""
        return output.model_dump()
