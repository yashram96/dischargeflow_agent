from typing import Dict, Any
from agents.base_agent import BaseAgent
from schemas.agent_schema import AgentOutputSchema
from utils.file_utils import read_json_file, format_evidence_path
import google.generativeai as genai
import json


class PharmacyAgent(BaseAgent):
    """
    Pharmacy verification agent that checks medication reconciliation,
    pending orders, drug interactions, and allergies using Gemini API.
    """
    
    def __init__(self, api_key: str = None):
        super().__init__("Pharmacy")
        
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def verify(self, patient_id: str, **kwargs) -> AgentOutputSchema:
        """Verify pharmacy requirements for discharge"""
        self.start_timer()
        
        # Load data files
        patient_data = self.load_patient_data()
        pharmacy_inventory = read_json_file("data/pharmacy_inventory.json")
        drug_interactions = read_json_file("data/drug_interaction_rules.json")
        
        self.add_checked_field("medications")
        self.add_checked_field("allergies")
        self.add_checked_field("active_orders")
        self.add_checked_field("drug_interactions")
        
        # Build prompt for Gemini
        prompt = self._build_verification_prompt(patient_data, pharmacy_inventory, drug_interactions)
        
        try:
            response = self.model.generate_content(prompt)
            result = self._parse_gemini_response(response.text)
            
            return self.create_output(
                noc=result["noc"],
                confidence=result["confidence"],
                issues=result["issues"],
                raw_response=result.get("raw_data", {})
            )
            
        except Exception as e:
            print(f"Gemini API error in PharmacyAgent: {e}")
            return self._fallback_verification(patient_data, pharmacy_inventory, drug_interactions)
    
    def _build_verification_prompt(self, patient_data: Dict, pharmacy_inventory: Dict, drug_interactions: Dict) -> str:
        """Build verification prompt for Gemini"""
        
        return f"""You are a Pharmacy Verification Agent for hospital discharge. 
Verify medication safety and readiness for patient discharge.

PATIENT MEDICATIONS:
{json.dumps(patient_data.get("Medications", {}), indent=2)}

PATIENT ALLERGIES:
{patient_data.get("Patient Information", {}).get("Allergies", "None")}

PHARMACY INVENTORY & ORDERS:
{json.dumps(pharmacy_inventory, indent=2)}

DRUG INTERACTION RULES:
{json.dumps(drug_interactions, indent=2)}

VERIFICATION TASKS:
1. Check for pending medication orders (status != "dispensed")
2. Verify no allergy-medication conflicts
3. Check for critical drug interactions
4. Verify discharge medication payment status
5. Check for duplicate medications

OUTPUT REQUIREMENTS:
Return ONLY valid JSON (no markdown, no explanation):
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
    "pending_orders": [],
    "allergy_conflicts": [],
    "interactions_found": []
  }}
}}

ISSUE CODES:
- PHARM_ORDER_PENDING: Medication order not dispensed
- PHARM_ALLERGY_CONFLICT: Medication conflicts with allergy
- PHARM_INTERACTION_CRITICAL: Critical drug interaction found
- PHARM_PAYMENT_PENDING: Discharge medication payment required
- PHARM_DUPLICATE: Duplicate medications detected

Set noc=false for critical issues (allergy conflicts, critical interactions, pending orders).
Set noc=true if all medications dispensed and no critical safety issues.
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
                issues.append(self.create_issue(
                    code=issue_data["code"],
                    title=issue_data["title"],
                    severity=issue_data["severity"],
                    message=issue_data["message"],
                    suggested_action=issue_data["suggested_action"],
                    evidence=issue_data.get("evidence", []),
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
    
    def _fallback_verification(self, patient_data: Dict, pharmacy_inventory: Dict, drug_interactions: Dict) -> AgentOutputSchema:
        """Fallback rule-based verification"""
        issues = []
        noc = True
        
        # Check for pending orders
        if pharmacy_inventory:
            for order in pharmacy_inventory.get("active_orders", []):
                if order.get("status") == "pending":
                    issues.append(self.create_issue(
                        code="PHARM_ORDER_PENDING",
                        title="Pending Medication Order",
                        severity="high",
                        message=f"Medication '{order.get('medication_name')}' order is pending dispense",
                        suggested_action="Dispense medication before discharge",
                        evidence=[format_evidence_path("data/pharmacy_inventory.json", f"active_orders[{order.get('order_id')}]")],
                        data={"order_id": order.get("order_id"), "medication": order.get("medication_name")}
                    ))
                    noc = False
            
            # Check payment for discharge medications
            total_cost = pharmacy_inventory.get("total_discharge_medication_cost", 0)
            if total_cost > 0:
                issues.append(self.create_issue(
                    code="PHARM_PAYMENT_PENDING",
                    title="Discharge Medication Payment Required",
                    severity="medium",
                    message=f"Patient needs to pay ₹{total_cost} for discharge medications",
                    suggested_action=f"Collect ₹{total_cost} from patient/family before discharge",
                    evidence=[format_evidence_path("data/pharmacy_inventory.json", "total_discharge_medication_cost")],
                    data={"amount": total_cost}
                ))
        
        # Check allergy conflicts
        allergies = patient_data.get("Patient Information", {}).get("Allergies", "").lower()
        if "liver" in allergies and drug_interactions:
            for contraindication in drug_interactions.get("allergy_contraindications", []):
                if "liver" in contraindication.get("allergy", "").lower():
                    # Check if patient is on contraindicated drugs
                    for med in patient_data.get("Medications", {}).get("Active Medications", []):
                        med_name = med.get("Name", "").lower()
                        if any(drug.lower() in med_name for drug in contraindication.get("contraindicated_drugs", [])):
                            issues.append(self.create_issue(
                                code="PHARM_ALLERGY_CONFLICT",
                                title="Allergy-Medication Conflict",
                                severity="critical",
                                message=f"Patient has liver complaint but is on {med.get('Name')} which may be contraindicated",
                                suggested_action="Consult physician for alternative medication",
                                evidence=[format_evidence_path("patient_data.json", "Medications.Active Medications")],
                                data={"medication": med.get("Name"), "allergy": allergies}
                            ))
                            noc = False
        
        return self.create_output(
            noc=noc,
            confidence=0.75,
            issues=issues,
            raw_response={"fallback": True}
        )
