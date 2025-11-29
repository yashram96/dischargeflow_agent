from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uvicorn
import os
import json
from pathlib import Path

from config import Config
from coordinator.workflow import DischargeWorkflow
from coordinator.escalation_manager import EscalationManager

# Initialize FastAPI app
app = FastAPI(
    title="Patient Discharge Automation API",
    description="API for verifying patient discharge readiness using multi-agent AI system",
    version="1.0.0"
)

# Initialize workflow components
# We initialize these lazily or globally depending on requirements
# For now, we'll create a new workflow instance per request to ensure clean state
# but we could cache the model configuration

class DischargeRequest(BaseModel):
    patient_id: str = Field(..., description="Patient identifier to verify")

class AlertsCount(BaseModel):
    total: int
    critical: int
    high: int
    medium: int
    low: int

class EscalationSummary(BaseModel):
    lab: int = 0
    billing: int = 0
    pharmacy: int = 0
    transport: int = 0
    insurance: int = 0
    general: int = 0

class DischargeResponse(BaseModel):
    patient_id: str
    status: str
    approved: bool
    timestamp: str
    summary: str
    alerts_count: AlertsCount
    escalations: Dict[str, int]
    details: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Validate configuration on startup"""
    try:
        Config.validate()
        print("✅ Configuration validated")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        # In a real app we might want to exit, but for dev we'll just log
        pass

@app.post("/api/v1/discharge/verify", response_model=DischargeResponse)
async def verify_discharge(request: DischargeRequest):
    """
    Trigger discharge verification workflow for a patient.
    """
    patient_id = request.patient_id
    print(f"🔍 Received discharge verification request for Patient ID: {patient_id}")
    
    try:
        # Initialize workflow
        workflow = DischargeWorkflow(api_key=Config.GEMINI_API_KEY)
        
        # Run workflow (synchronous for now, could be made async with background tasks for long running)
        final_state = workflow.run(patient_id)
        
        # Calculate alert counts from aggregated issues
        issues = final_state.get("aggregated_issues", [])
        
        alerts_count = {
            "total": len(issues),
            "critical": len([i for i in issues if i.get("severity") == "critical"]),
            "high": len([i for i in issues if i.get("severity") == "high"]),
            "medium": len([i for i in issues if i.get("severity") == "medium"]),
            "low": len([i for i in issues if i.get("severity") == "low"])
        }
        
        # Calculate escalations by department
        # We can reuse EscalationManager logic or just count based on prefixes
        escalation_manager = EscalationManager()
        escalations = {
            "Lab Portal": 0,
            "Pharmacy Portal": 0,
            "Billing Portal": 0,
            "Transport Services": 0,
            "Insurance Desk": 0,
            "General Operations": 0
        }
        
        for issue in issues:
            dept = escalation_manager._map_issue_to_department(issue.get("code", ""))
            if dept in escalations:
                escalations[dept] += 1
            else:
                escalations["General Operations"] += 1
                
        # Simplify escalation keys for API response
        simple_escalations = {
            "lab": escalations.get("Lab Portal", 0),
            "pharmacy": escalations.get("Pharmacy Portal", 0),
            "billing": escalations.get("Billing Portal", 0),
            "transport": escalations.get("Transport Services", 0),
            "insurance": escalations.get("Insurance Desk", 0),
            "general": escalations.get("General Operations", 0)
        }
        
        # Construct response
        return DischargeResponse(
            patient_id=patient_id,
            status=final_state["final_decision"],
            approved=final_state["approved"],
            timestamp=final_state["timestamp"],
            summary=final_state["discharge_summary"].get("plain_text", "No summary available"),
            alerts_count=AlertsCount(**alerts_count),
            escalations=simple_escalations,
            details={
                "approved_by": final_state["approved_by"],
                "blocked_by": final_state["blocked_by"],
                "suggested_auto_resolutions": final_state["suggested_auto_resolutions"]
            }
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
