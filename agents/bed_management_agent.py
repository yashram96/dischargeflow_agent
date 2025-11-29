from typing import Dict, Any
from agents.base_agent import BaseAgent
from schemas.agent_schema import AgentOutputSchema
from utils.file_utils import read_json_file, format_evidence_path
import google.generativeai as genai
import json
from config import Config

class BedManagementAgent(BaseAgent):
    """
    Bed Management verification agent that checks billing status,
    deposit sufficiency, and housekeeping schedule using Gemini API.
    """
    
    def __init__(self, api_key: str = None):
        super().__init__("Bed Management")
        
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        
    def verify(self, patient_id: str, **kwargs) -> AgentOutputSchema:
        """Verify bed and billing requirements"""
        self.start_timer()
        
        # Load data files
        patient_data = self.load_patient_data()
        billing_snapshot = read_json_file("data/billing_snapshot.json")
        housekeeping_schedule = read_json_file("data/housekeeping_schedule.json")
        
        self.add_checked_field("billing_status")
        self.add_checked_field("deposit_paid")
        self.add_checked_field("housekeeping_schedule")
        
        # Build prompt for Gemini
        prompt = self._build_verification_prompt(patient_data, billing_snapshot, housekeeping_schedule)
        
        try:
            # print("  Calling Gemini API for bed management verification...")
            
            generation_config = {
                "temperature": 0.1,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json"
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            print("  Gemini API response received, parsing...")
            result = self._parse_gemini_response(response.text)
            print(f"  ✓ Bed management verification complete (NOC: {result['noc']})")
            
            return self.create_output(
                noc=result["noc"],
                confidence=result["confidence"],
                issues=result["issues"],
                raw_response=result.get("raw_data", {})
            )
            
        except Exception as e:
            print(f"  ✗ Gemini API error: {type(e).__name__}: {str(e)}")
            print(f"  → Falling back to rule-based verification...")
            return self._fallback_verification(patient_data, billing_snapshot, housekeeping_schedule)
    
    def _build_verification_prompt(self, patient_data: Dict, billing_snapshot: Dict, housekeeping_schedule: Dict) -> str:
        """Build verification prompt for Gemini"""
        
        return f"""You are a Bed Management Verification Agent for hospital discharge.
Verify billing completion, deposit sufficiency, and bed turnover readiness.

PATIENT BILLING DATA:
{json.dumps(patient_data.get("Billing", {}), indent=2)}

BILLING SNAPSHOT:
{json.dumps(billing_snapshot, indent=2)}

HOUSEKEEPING SCHEDULE:
{json.dumps(housekeeping_schedule, indent=2)}

VERIFICATION TASKS:
1. Check if final invoice is generated (invoice_generated should be true)
2. Verify deposit sufficiency:
   - Compare deposit_paid vs required_before_discharge
   - Check if patient_balance is acceptable
3. Verify housekeeping schedule exists for bed turnover
4. Check for pending billing adjustments

OUTPUT REQUIREMENTS:
Return ONLY valid JSON (no markdown):
{{
  "noc": true or false,
  "confidence": 0.0 to 1.0,
  "issues": [
    {{
      "code": "ISSUE_CODE",
      "title": "Short title",
      "severity": "low|medium|high|critical",
      "message": "Detailed explanation",
      "suggested_action": "What to do",
      "evidence": ["file_path#reference"],
      "data": {{"key": "value"}}
    }}
  ],
  "raw_data": {{
    "invoice_generated": true or false,
    "deposit_sufficient": true or false,
    "housekeeping_scheduled": true or false,
    "shortfall_amount": 0
  }}
}}

ISSUE CODES:
- BED_INVOICE_PENDING: Final invoice not generated
- BED_DEPOSIT_SHORTFALL: Insufficient deposit paid
- BED_CLEANUP_DELAY: Housekeeping not scheduled timely
- BED_BILLING_ADJUSTMENT: Pending billing adjustments

Set noc=false if invoice not generated or significant deposit shortfall.
Set noc=true if billing complete and deposit sufficient (or refund due).
"""
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini response"""
        try:
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            
            issues = []
            for issue_data in result.get("issues", []):
                # Normalize severity
                severity = issue_data.get("severity", "medium").lower()
                if severity not in ["low", "medium", "high", "critical"]:
                    severity = "medium"
                
                # Normalize evidence
                evidence = issue_data.get("evidence", [])
                if isinstance(evidence, dict):
                    evidence = [f"{k}: {v}" for k, v in evidence.items()]
                elif isinstance(evidence, str):
                    evidence = [evidence]

                issues.append(self.create_issue(
                    code=issue_data["code"],
                    title=issue_data["title"],
                    severity=severity,
                    message=issue_data["message"],
                    suggested_action=issue_data["suggested_action"],
                    evidence=evidence,
                    data=issue_data.get("data", {})
                ))
            
            return {
                "noc": result["noc"],
                "confidence": result["confidence"],
                "issues": issues,
                "raw_data": result.get("raw_data", {})
            }
            
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            raise
    
    def _fallback_verification(self, patient_data: Dict, billing_snapshot: Dict, housekeeping_schedule: Dict) -> AgentOutputSchema:
        """Fallback rule-based verification"""
        issues = []
        noc = True
        
        if not billing_snapshot:
            issues.append(self.create_issue(
                code="BED_INVOICE_PENDING",
                title="Billing Data Missing",
                severity="high",
                message="Unable to verify billing status - billing snapshot not available",
                suggested_action="Generate final invoice through billing system",
                evidence=["data/billing_snapshot.json"]
            ))
            noc = False
        else:
            # Check invoice generation
            invoice_status = billing_snapshot.get("invoice_status", {})
            if not invoice_status.get("invoice_generated"):
                issues.append(self.create_issue(
                    code="BED_INVOICE_PENDING",
                    title="Final Invoice Not Generated",
                    severity="high",
                    message=f"Invoice status: {invoice_status.get('status', 'pending')}",
                    suggested_action="Generate final invoice via Billing UI before discharge",
                    evidence=[format_evidence_path("data/billing_snapshot.json", "invoice_status.invoice_generated")],
                    data={"status": invoice_status.get("status")}
                ))
                noc = False
            
            # Check deposit
            payments = billing_snapshot.get("payments", {})
            required = payments.get("required_before_discharge", 0)
            deposit_analysis = billing_snapshot.get("deposit_analysis", {})
            
            if required > 0:
                issues.append(self.create_issue(
                    code="BED_DEPOSIT_SHORTFALL",
                    title="Payment Required Before Discharge",
                    severity="high",
                    message=f"Patient needs to pay ₹{required} before discharge",
                    suggested_action=f"Collect ₹{required} from patient/family",
                    evidence=[format_evidence_path("data/billing_snapshot.json", "payments.required_before_discharge")],
                    data={"amount": required}
                ))
                noc = False
            elif deposit_analysis.get("refund_due", 0) > 0:
                issues.append(self.create_issue(
                    code="BED_REFUND_DUE",
                    title="Deposit Refund Due",
                    severity="low",
                    message=f"Refund of ₹{deposit_analysis.get('refund_due')} due to patient",
                    suggested_action="Process refund after final invoice generation",
                    evidence=[format_evidence_path("data/billing_snapshot.json", "deposit_analysis.refund_due")],
                    data={"refund_amount": deposit_analysis.get("refund_due")}
                ))
            
            # Check housekeeping
            if housekeeping_schedule and housekeeping_schedule.get("cleaning_schedule"):
                # Housekeeping is scheduled - good
                pass
            else:
                issues.append(self.create_issue(
                    code="BED_CLEANUP_DELAY",
                    title="Housekeeping Not Scheduled",
                    severity="medium",
                    message="Bed cleaning schedule not found",
                    suggested_action="Schedule terminal cleaning for bed turnover",
                    evidence=["data/housekeeping_schedule.json"]
                ))
        
        return self.create_output(
            noc=noc,
            confidence=0.75,
            issues=issues,
            raw_response={"fallback": True}
        )
