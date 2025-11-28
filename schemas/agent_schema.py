from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime


class IssueSchema(BaseModel):
    """Schema for an individual issue found by an agent"""
    code: str = Field(..., description="Unique issue code (e.g., 'INS_PREAUTH_MISSING')")
    title: str = Field(..., description="Short title of the issue")
    severity: Literal["low", "medium", "high", "critical"] = Field(..., description="Issue severity level")
    message: str = Field(..., description="Human-readable explanation of the issue")
    suggested_action: str = Field(..., description="What the coordinator/staff should do")
    evidence: List[str] = Field(default_factory=list, description="File paths or snippets supporting the issue")
    data: Dict[str, Any] = Field(default_factory=dict, description="Raw data snippet that supports the issue")


class AgentMetadata(BaseModel):
    """Metadata about agent execution"""
    checked_fields: List[str] = Field(default_factory=list, description="Fields that were checked")
    time_ms: float = Field(..., description="Execution time in milliseconds")
    raw_response: Dict[str, Any] = Field(default_factory=dict, description="Optional raw data from checks")


class AgentOutputSchema(BaseModel):
    """Standard output schema for all agents"""
    agent: str = Field(..., description="Name of the agent (e.g., 'Insurance', 'Pharmacy')")
    noc: bool = Field(..., description="No Objection Certificate granted (True) or not (False)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    issues: List[IssueSchema] = Field(default_factory=list, description="List of issues found")
    meta: AgentMetadata = Field(..., description="Metadata about the agent execution")


class CoordinatorDecisionSchema(BaseModel):
    """Schema for coordinator final decision"""
    patient_id: str = Field(..., description="Patient identifier")
    final_decision: Literal["APPROVE", "HOLD", "PENDING_AUTO_RESOLUTION"] = Field(..., description="Final discharge decision")
    approved: bool = Field(..., description="Whether discharge is approved")
    approved_by: List[str] = Field(default_factory=list, description="List of agents that granted NOC")
    blocked_by: List[str] = Field(default_factory=list, description="List of agents blocking discharge")
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="Aggregated issues from all agents")
    suggested_auto_resolutions: List[Dict[str, Any]] = Field(default_factory=list, description="Suggested automated resolutions")
    discharge_summary: Dict[str, str] = Field(..., description="Discharge summary with plain_text and for_medical_record")
    files_written: List[str] = Field(default_factory=list, description="List of files written during coordination")
    timestamp: str = Field(..., description="ISO8601 timestamp of decision")
