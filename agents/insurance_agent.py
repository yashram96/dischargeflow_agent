from typing import Dict, Any
from agents.base_agent import BaseAgent
from schemas.agent_schema import AgentOutputSchema
from utils.file_utils import read_json_file, format_evidence_path
import google.generativeai as genai
import json
import os


from config import Config

class InsuranceAgent(BaseAgent):
    """
    Insurance verification agent that checks policy status, coverage, 
    pre-authorization, and limits using Gemini API.
    """
    
    def __init__(self, api_key: str = None):
        super().__init__("Insurance")
        
        # Configure Gemini API
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        
    def verify(self, patient_id: str, **kwargs) -> AgentOutputSchema:
        """
        Verify insurance coverage and authorization.
        
        Args:
            patient_id: Patient identifier
            **kwargs: Additional parameters
            
        Returns:
            AgentOutputSchema with verification results
        """
        self.start_timer()
        
        # Load data files
        patient_data = self.load_patient_data()
        insurer_records = read_json_file("data/insurer_records.json")
        
        # Read insurance policy text
        try:
            with open("insurance_policy.txt", "r", encoding="utf-8") as f:
                policy_text = f.read()
        except:
            policy_text = "Policy file not available"
        
        self.add_checked_field("policy_number")
        self.add_checked_field("coverage_limits")
        self.add_checked_field("pre_authorization")
        
        # Build prompt for Gemini
        prompt = self._build_verification_prompt(patient_data, insurer_records, policy_text)
        
        # Try Gemini API
        try:
            # print("  Calling Gemini API for insurance verification...")
            
            # Set generation config
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json"
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            print("  Gemini API response received, parsing...")
            
            # Debug: Print full response details
            # print(f"  DEBUG - Response object: {response}")
            # if hasattr(response, 'candidates') and response.candidates:
            #     print(f"  DEBUG - Finish reason: {response.candidates[0].finish_reason}")
            
            # Check if response has text
            if not response or not response.text:
                print("  ✗ Gemini API returned empty response")
                print("  → Falling back to rule-based verification...")
                return self._fallback_verification(patient_data, insurer_records)
            
            # print(f"  DEBUG - Response text: {response.text[:200]}...")
            result = self._parse_gemini_response(response.text)
            
            print(f"  ✓ Insurance verification complete (NOC: {result['noc']})")
            
            return self.create_output(
                noc=result["noc"],
                confidence=result["confidence"],
                issues=result["issues"],
                raw_response=result.get("raw_data", {})
            )
            
        except Exception as e:
            print(f"  ✗ Gemini API error: {type(e).__name__}: {str(e)}")
            print(f"  → Falling back to rule-based verification...")
            return self._fallback_verification(patient_data, insurer_records)
    
    def _build_verification_prompt(self, patient_data: Dict, insurer_records: Dict, policy_text: str) -> str:
        """Build the prompt for Gemini API"""
        
        insurance_details = patient_data.get("Insurance Details", {})
        billing = patient_data.get("Billing", {})
        
        return f"""Analyze this insurance verification case and provide your assessment in JSON format.

Patient's Insurance Information:
- Policy: {insurance_details.get("Provider Information", {}).get("Policy Number", "N/A")}
- Provider: {insurance_details.get("Provider Information", {}).get("Provider", "N/A")}
- Policy Status: {insurer_records.get("policy_details", {}).get("policy_status", "unknown")}
- Coverage Type: {insurance_details.get("Coverage Details", {}).get("Coverage Type", "N/A")}
- Annual Limit: {insurance_details.get("Coverage Details", {}).get("Coverage Limit", "N/A")}
- Pre-authorization Required: {insurance_details.get("Coverage Details", {}).get("Pre-authorization", "N/A")}

Financial Details:
- Total Hospital Bill: {billing.get("Total Cost", "N/A")}
- Amount Covered by Insurance: {billing.get("Insurance Covered", "N/A")}
- Patient's Balance: {billing.get("Patient Balance", "N/A")}

Pre-Authorization Records:
{json.dumps(insurer_records.get("pre_authorization_records", []), indent=2)[:600]}

Coverage Limits Available:
{json.dumps(insurer_records.get("coverage_limits", {}), indent=2)[:400]}

Based on this information, assess whether the patient can be discharged from an insurance perspective. Check:
1. Is the policy currently active?
2. If pre-authorization was required, has it been approved?
3. Are the coverage limits sufficient for the total bill?
4. What is the patient's financial responsibility?

Provide your analysis as a JSON object with these fields:
- noc: boolean (true if insurance clears discharge, false if there are blocking issues)
- confidence: number between 0 and 1
- issues: array of any problems found (each with code, title, severity, message, suggested_action, evidence, data)
- raw_data: object with policy_status, preauth_status, coverage_sufficient

Use these issue codes when needed: INS_POLICY_EXPIRED, INS_PREAUTH_MISSING, INS_LIMITS_EXCEEDED, INS_PARTIAL_COVERAGE
"""
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini API response"""
        try:
            # Remove markdown code blocks if present
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            
            # Convert issues to IssueSchema objects
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
            print(f"Response: {response_text}")
            raise
    
    def _fallback_verification(self, patient_data: Dict, insurer_records: Dict) -> AgentOutputSchema:
        """Fallback rule-based verification if Gemini fails"""
        issues = []
        noc = True
        
        if not insurer_records:
            issues.append(self.create_issue(
                code="INS_DATA_MISSING",
                title="Insurance Records Missing",
                severity="high",
                message="Unable to verify insurance - insurer records not available",
                suggested_action="Contact insurance desk to verify policy manually",
                evidence=["data/insurer_records.json"]
            ))
            noc = False
        else:
            # Check policy status
            policy_status = insurer_records.get("policy_details", {}).get("policy_status")
            if policy_status != "active":
                issues.append(self.create_issue(
                    code="INS_POLICY_EXPIRED",
                    title="Policy Not Active",
                    severity="critical",
                    message=f"Insurance policy status: {policy_status}",
                    suggested_action="Contact insurance provider to reactivate policy",
                    evidence=[format_evidence_path("data/insurer_records.json", "policy_details.policy_status")]
                ))
                noc = False
            
            # Check pre-authorization
            preauth_records = insurer_records.get("pre_authorization_records", [])
            if not preauth_records or preauth_records[0].get("status") != "approved":
                issues.append(self.create_issue(
                    code="INS_PREAUTH_MISSING",
                    title="Pre-Authorization Missing",
                    severity="high",
                    message="No approved pre-authorization found for this admission",
                    suggested_action="Submit pre-authorization request to insurance",
                    evidence=[format_evidence_path("data/insurer_records.json", "pre_authorization_records")]
                ))
                noc = False
        
        return self.create_output(
            noc=noc,
            confidence=0.7,
            issues=issues,
            raw_response={"fallback": True}
        )
