from typing import Dict, Any, List
from datetime import datetime
import google.generativeai as genai
import json

from schemas.agent_schema import CoordinatorDecisionSchema
from coordinator.state_manager import StateManager
from coordinator.escalation_manager import EscalationManager
from utils.file_utils import get_iso_timestamp
from config import Config

class CoordinatorAgent:
    """
    Coordinator agent that orchestrates all 5 agents and makes final discharge decision.
    Uses Gemini API for intelligent decision synthesis.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize coordinator agent.
        
        Args:
            api_key: Gemini API key
        """
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        self.state_manager = StateManager()
        self.escalation_manager = EscalationManager()
    
    def coordinate(
        self,
        patient_id: str,
        agent_outputs: Dict[str, Dict[str, Any]]
    ) -> CoordinatorDecisionSchema:
        """
        Coordinate all agent outputs and make final decision.
        
        Args:
            patient_id: Patient identifier
            agent_outputs: Dictionary of agent outputs
                {
                    "Insurance": {...},
                    "Pharmacy": {...},
                    "Ambulance": {...},
                    "Bed Management": {...},
                    "Lab": {...}
                }
        
        Returns:
            CoordinatorDecisionSchema with final decision
        """
        # Aggregate all issues
        all_issues = []
        approved_by = []
        blocked_by = []
        
        for agent_name, output in agent_outputs.items():
            if output.get("noc"):
                approved_by.append(agent_name)
            
            for issue in output.get("issues", []):
                issue_with_agent = issue.copy()
                issue_with_agent["agent"] = agent_name
                all_issues.append(issue_with_agent)
                
                # Track blocking agents
                if issue.get("severity") in ["high", "critical"] and agent_name not in blocked_by:
                    blocked_by.append(agent_name)
        
        # Apply decision rules
        final_decision, approved = self._apply_decision_rules(all_issues, agent_outputs)
        
        # Generate discharge summary
        discharge_summary = self._generate_discharge_summary(
            patient_id, agent_outputs, all_issues, final_decision
        )
        
        # Generate auto-resolution suggestions
        suggested_auto_resolutions = self._generate_auto_resolutions(all_issues, final_decision)
        
        # Save state to files
        files_written = []
        
        # Save discharge state
        state_file = self.state_manager.save_discharge_state(
            patient_id=patient_id,
            status=final_decision.lower().replace("_", " "),
            agents_output=agent_outputs,
            issues=all_issues,
            final_decision=final_decision,
            approved_by=approved_by,
            blocked_by=blocked_by
        )
        files_written.append(state_file)
        
        # Save audit log
        next_steps = [res.get("action", "") for res in suggested_auto_resolutions]
        audit_file = self.state_manager.append_audit_log(
            patient_id=patient_id,
            final_decision=final_decision,
            issues=all_issues,
            recommended_next_steps=next_steps
        )
        files_written.append(audit_file)
        
        # Generate escalation alerts for departments
        print("  Generating escalation alerts for departments...")
        escalation_files = self.escalation_manager.create_escalations(
            patient_id=patient_id,
            issues=all_issues,
            final_decision=final_decision
        )
        files_written.extend(escalation_files)
        print(f"  ✓ Generated {len(escalation_files)} escalation alert files")
        
        # Create coordinator decision
        return CoordinatorDecisionSchema(
            patient_id=patient_id,
            final_decision=final_decision,
            approved=approved,
            approved_by=approved_by,
            blocked_by=blocked_by,
            issues=all_issues,
            suggested_auto_resolutions=suggested_auto_resolutions,
            discharge_summary=discharge_summary,
            files_written=files_written,
            timestamp=get_iso_timestamp()
        )
    
    def _apply_decision_rules(
        self,
        all_issues: List[Dict[str, Any]],
        agent_outputs: Dict[str, Dict[str, Any]]
    ) -> tuple[str, bool]:
        """
        Apply decision rules to determine final decision.
        
        Returns:
            Tuple of (final_decision, approved)
        """
        # Rule 1: Any critical issue -> HOLD
        critical_issues = [i for i in all_issues if i.get("severity") == "critical"]
        if critical_issues:
            return ("HOLD", False)
        
        # Rule 2: Any high severity issue -> HOLD
        high_issues = [i for i in all_issues if i.get("severity") == "high"]
        if high_issues:
            return ("HOLD", False)
        
        # Rule 3: All agents granted NOC -> APPROVE
        all_noc = all(output.get("noc", False) for output in agent_outputs.values())
        if all_noc:
            return ("APPROVE", True)
        
        # Rule 4: Only medium/low issues -> PENDING_AUTO_RESOLUTION
        return ("PENDING_AUTO_RESOLUTION", False)
    
    def _generate_discharge_summary(
        self,
        patient_id: str,
        agent_outputs: Dict[str, Dict[str, Any]],
        all_issues: List[Dict[str, Any]],
        final_decision: str
    ) -> Dict[str, str]:
        """Generate discharge summary using Gemini API"""
        
        prompt = f"""You are a Discharge Coordinator. Generate a discharge summary based on agent verifications.

PATIENT ID: {patient_id}

AGENT OUTPUTS:
{json.dumps(agent_outputs, indent=2)[:1000]}

ALL ISSUES:
{json.dumps(all_issues, indent=2)[:1000]}

FINAL DECISION: {final_decision}

Generate TWO summaries:

1. PLAIN TEXT (for patient/family):
   - Simple, non-technical language
   - If APPROVED: explain discharge is ready, next steps
   - If HOLD: explain what's blocking and what needs to happen
   - If PENDING: explain minor issues being resolved

2. FOR MEDICAL RECORD (for clinicians):
   - Professional medical summary
   - Include key findings from each agent
   - List any pending items or follow-up required
   - Reference specific issue codes

Return ONLY valid JSON (no markdown):
{{
  "plain_text": "Patient-friendly summary paragraph",
  "for_medical_record": "Professional clinical summary"
}}
"""
        
        try:
            print("  Generating discharge summary with Gemini API...")
            response = self.model.generate_content(
                prompt,
                request_options={"timeout": 30}
            )
            cleaned = response.text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            summary = json.loads(cleaned)
            print("  ✓ Discharge summary generated")
            return summary
        except Exception as e:
            print(f"  ✗ Gemini API error: {type(e).__name__}: {str(e)}")
            print(f"  → Using fallback summary generation...")
            return self._fallback_summary(final_decision, all_issues)
    
    def _fallback_summary(self, final_decision: str, all_issues: List[Dict[str, Any]]) -> Dict[str, str]:
        """Fallback summary generation"""
        if final_decision == "APPROVE":
            plain_text = "Patient discharge has been approved. All verification checks passed successfully. Please proceed with discharge procedures."
            medical = "Discharge approved. All agents (Insurance, Pharmacy, Ambulance, Bed Management, Lab) granted NOC. No blocking issues identified."
        elif final_decision == "HOLD":
            issue_summary = ", ".join([f"{i.get('agent')}: {i.get('title')}" for i in all_issues[:3]])
            plain_text = f"Discharge is on hold due to pending issues: {issue_summary}. Please contact hospital staff for details."
            medical = f"Discharge HOLD. Critical/high severity issues identified: {issue_summary}. Resolution required before discharge."
        else:
            plain_text = "Discharge is pending minor issue resolution. Hospital staff is working to resolve these items."
            medical = f"Discharge pending auto-resolution. {len(all_issues)} medium/low severity issues identified. Staff action required."
        
        return {
            "plain_text": plain_text,
            "for_medical_record": medical
        }
    
    def _generate_auto_resolutions(
        self,
        all_issues: List[Dict[str, Any]],
        final_decision: str
    ) -> List[Dict[str, Any]]:
        """Generate suggested auto-resolution steps"""
        
        if final_decision == "APPROVE":
            return []
        
        resolutions = []
        
        for issue in all_issues:
            if issue.get("severity") in ["medium", "low"]:
                resolutions.append({
                    "action": issue.get("suggested_action", ""),
                    "details": {
                        "agent": issue.get("agent"),
                        "code": issue.get("code"),
                        "severity": issue.get("severity"),
                        "data": issue.get("data", {})
                    }
                })
        
        return resolutions
