from typing import Dict, Any
from agents.base_agent import BaseAgent
from schemas.agent_schema import AgentOutputSchema
from utils.file_utils import read_json_file, format_evidence_path
import google.generativeai as genai
import json
from config import Config

class PharmacyAgent(BaseAgent):
    """
    Pharmacy verification agent that checks medication reconciliation,
    pending orders, drug interactions, and allergies using Gemini API.
    """
    
    def __init__(self, api_key: str = None):
        super().__init__("Pharmacy")
        
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        
    def verify(self, patient_id: str, **kwargs) -> AgentOutputSchema:
        """Verify pharmacy requirements for discharge"""
        self.start_timer()
        
        # Load data files
        patient_data = self.load_patient_data(patient_id)
        all_pharmacy_inventory = read_json_file("data/pharmacy_inventory.json")
        pharmacy_inventory = self.get_patient_record(all_pharmacy_inventory, patient_id)
        
        drug_interactions = read_json_file("data/drug_interaction_rules.json")
        
        self.add_checked_field("medications")
        self.add_checked_field("allergies")
        self.add_checked_field("active_orders")
        self.add_checked_field("drug_interactions")
        
        # Build prompt for Gemini
        prompt = self._build_verification_prompt(patient_data, pharmacy_inventory, drug_interactions)
        
        try:
            # print("  Calling Gemini API for pharmacy verification...")
            
            generation_config = {
                "temperature": 0.1,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json"
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # print("  Gemini API response received, parsing...")
            
            # Debug response
            # print(f"  DEBUG - Response: {response}")
            # if hasattr(response, 'candidates') and response.candidates:
            #     print(f"  DEBUG - Finish reason: {response.candidates[0].finish_reason}")
            
            # Check if response has text
            if not response or not response.text:
                print("  ✗ Gemini API returned empty response")
                print("  → Falling back to rule-based verification...")
                return self._fallback_verification(patient_data, pharmacy_inventory, drug_interactions)
            
            # print(f"  DEBUG - Response text: {response.text[:200]}...")
            result = self._parse_gemini_response(response.text)
            
            print(f"  ✓ Pharmacy verification complete (NOC: {result['noc']})")
            
            return self.create_output(
                noc=result["noc"],
                confidence=result["confidence"],
                issues=result["issues"],
                raw_response=result.get("raw_data", {})
            )
            
        except Exception as e:
            print(f"  ✗ Gemini API error: {type(e).__name__}: {str(e)}")
            print(f"  → Falling back to rule-based verification...")
            return self._fallback_verification(patient_data, pharmacy_inventory, drug_interactions)
    
    def _build_verification_prompt(self, patient_data: Dict, pharmacy_inventory: Dict, drug_interactions: Dict) -> str:
        """Build verification prompt for Gemini"""
        
        medications = patient_data.get("Medications", {}).get("Active Medications", [])
        allergies = patient_data.get("Patient Information", {}).get("Allergies", "None")
        
        return f"""Review this patient's medication status for hospital discharge clearance.

Current Medications:
{json.dumps(medications, indent=2)[:700]}

Known Allergies: {allergies}

Pharmacy Order Status:
{json.dumps(pharmacy_inventory.get("active_orders", []), indent=2)[:700]}

Discharge Prescriptions:
{json.dumps(pharmacy_inventory.get("discharge_medications", []), indent=2)[:500]}

Please verify:
1. Have all medication orders been filled and dispensed?
2. Are there any conflicts between medications and patient allergies?
3. Are there duplicate medications prescribed?
4. Is payment cleared for discharge medications?

Respond with a JSON assessment containing:
- noc: true if pharmacy clears discharge, false if issues block it
- confidence: your confidence level (0-1)
- issues: list any problems (with code, title, severity, message, suggested_action)
- raw_data: summary with pending_orders, allergy_conflicts, interactions_found arrays

Common issue codes: PHARM_ORDER_PENDING, PHARM_ALLERGY_CONFLICT, PHARM_PAYMENT_PENDING, PHARM_DUPLICATE
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
