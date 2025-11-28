from typing import Dict, Any
from agents.base_agent import BaseAgent
from schemas.agent_schema import AgentOutputSchema
from utils.file_utils import read_json_file, format_evidence_path
import google.generativeai as genai
import json
import os


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
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
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
        
        try:
            # Call Gemini API
            response = self.model.generate_content(prompt)
            result = self._parse_gemini_response(response.text)
            
            return self.create_output(
                noc=result["noc"],
                confidence=result["confidence"],
                issues=result["issues"],
                raw_response=result.get("raw_data", {})
            )
            
        except Exception as e:
            # Fallback to rule-based verification
            return self._fallback_verification(patient_data, insurer_records)
    
    def _build_verification_prompt(self, patient_data: Dict, insurer_records: Dict, policy_text: str) -> str:
        """Build the prompt for Gemini API"""
        
        return f"""You are an Insurance Verification Agent for a hospital discharge system. 
Analyze the following data and determine if the patient's insurance provides No Objection Certificate (NOC) for discharge.

PATIENT DATA:
{json.dumps(patient_data.get("Insurance Details", {}), indent=2)}

BILLING DATA:
{json.dumps(patient_data.get("Billing", {}), indent=2)}

INSURER RECORDS:
{json.dumps(insurer_records, indent=2)}

INSURANCE POLICY TERMS:
{policy_text[:2000]}  # Limit policy text length

VERIFICATION TASKS:
1. Check if policy is active on admission date
2. Verify pre-authorization status for procedures
3. Check coverage limits against estimated bill
4. Identify any exclusions or conditions
5. Calculate patient responsibility (co-pay, deductible)

OUTPUT REQUIREMENTS:
Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
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
    "policy_status": "active|expired",
    "preauth_status": "approved|missing|pending",
    "coverage_sufficient": true or false
  }}
}}

ISSUE CODES TO USE:
- INS_POLICY_EXPIRED: Policy not active
- INS_PREAUTH_MISSING: Pre-authorization missing
- INS_LIMITS_EXCEEDED: Coverage limits insufficient
- INS_PARTIAL_COVERAGE: Partial coverage with patient responsibility
- INS_EXCLUSION_FOUND: Procedure excluded from coverage

Set noc=false for critical issues (expired policy, missing preauth, insufficient limits).
Set noc=true if all checks pass or only minor issues (co-pay collection needed).
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
