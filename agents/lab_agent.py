from typing import Dict, Any
from agents.base_agent import BaseAgent
from schemas.agent_schema import AgentOutputSchema
from utils.file_utils import read_json_file, format_evidence_path
import google.generativeai as genai
import json
from config import Config

class LabAgent(BaseAgent):
    """
    Lab verification agent that confirms test completion and
    checks for critical values using Gemini API.
    """
    
    def __init__(self, api_key: str = None):
        super().__init__("Lab")
        
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        
    def verify(self, patient_id: str, **kwargs) -> AgentOutputSchema:
        """Verify lab test completion and results"""
        self.start_timer()
        
        # Load data files
        patient_data = self.load_patient_data()
        lab_results = read_json_file("data/lab_results.json")
        
        self.add_checked_field("lab_tests")
        self.add_checked_field("test_results")
        self.add_checked_field("critical_values")
        
        # Build prompt for Gemini
        prompt = self._build_verification_prompt(patient_data, lab_results)
        
        try:
            # print("  Calling Gemini API for lab verification...")
            
            generation_config = {
                "temperature": 0.1,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json"
            }
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Check if response has text
            if not response or not response.text:
                print("  ✗ Gemini API returned empty response")
                print("  → Falling back to rule-based verification...")
                return self._fallback_verification(patient_data, lab_results)
            
            print("  Gemini API response received, parsing...")
            result = self._parse_gemini_response(response.text)
            print(f"  ✓ Lab verification complete (NOC: {result['noc']})")
            
            return self.create_output(
                noc=result["noc"],
                confidence=result["confidence"],
                issues=result["issues"],
                raw_response=result.get("raw_data", {})
            )
            
        except Exception as e:
            print(f"  ✗ Gemini API error: {type(e).__name__}: {str(e)}")
            print(f"  → Falling back to rule-based verification...")
            return self._fallback_verification(patient_data, lab_results)
    
    def _build_verification_prompt(self, patient_data: Dict, lab_results: Dict) -> str:
        """Build verification prompt for Gemini"""
        
        return f"""You are a Lab Verification Agent for hospital discharge.
Verify all required lab tests are completed and results are within safe ranges.

PATIENT LAB TESTS (from patient record):
{json.dumps(patient_data.get("Lab Tests & Results", {}), indent=2)}

LAB RESULTS DATABASE:
{json.dumps(lab_results, indent=2)}

VERIFICATION TASKS:
1. Check all required tests are completed (status = "completed")
2. Identify any pending tests
3. Check for critical values (flag = "critical")
4. Verify results are within acceptable ranges for discharge
5. Ensure all tests completed before discharge time

CRITICAL VALUE THRESHOLDS (general):
- Hemoglobin: <7.0 g/dL is critical
- RBC: <3.0 is critical
- Platelets: >450 or <50 is critical
- WBC: >20 or <2 is critical

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
    "pending_tests": [],
    "critical_values": [],
    "all_tests_complete": true or false
  }}
}}

ISSUE CODES:
- LAB_PENDING: Required test not completed
- LAB_CRITICAL_VALUE: Test result shows critical value
- LAB_DATA_MISSING: Lab results not available

Set noc=false if any tests pending or critical values found.
Set noc=true if all tests complete and values acceptable for discharge.
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
    
    def _fallback_verification(self, patient_data: Dict, lab_results: Dict) -> AgentOutputSchema:
        """Fallback rule-based verification"""
        issues = []
        noc = True
        
        if not lab_results:
            issues.append(self.create_issue(
                code="LAB_DATA_MISSING",
                title="Lab Results Not Available",
                severity="high",
                message="Unable to verify lab tests - results database not available",
                suggested_action="Retrieve lab results from laboratory system",
                evidence=["data/lab_results.json"]
            ))
            noc = False
        else:
            # Check each required test
            required_tests = lab_results.get("required_tests", [])
            results = lab_results.get("results", [])
            
            for required_test in required_tests:
                # Find matching result
                matching_result = None
                for result in results:
                    if result.get("test_name") == required_test:
                        matching_result = result
                        break
                
                if not matching_result:
                    issues.append(self.create_issue(
                        code="LAB_PENDING",
                        title=f"Missing Test: {required_test}",
                        severity="high",
                        message=f"Required test '{required_test}' not found in results",
                        suggested_action="Complete the required test before discharge",
                        evidence=[format_evidence_path("data/lab_results.json", "required_tests")]
                    ))
                    noc = False
                elif matching_result.get("status") == "pending":
                    issues.append(self.create_issue(
                        code="LAB_PENDING",
                        title=f"Pending Test: {required_test}",
                        severity="high",
                        message=f"Test '{required_test}' is still pending",
                        suggested_action="Wait for test completion or expedite processing",
                        evidence=[format_evidence_path("data/lab_results.json", f"results[{matching_result.get('test_id')}]")],
                        data={"test_id": matching_result.get("test_id")}
                    ))
                    noc = False
                else:
                    # Check for critical values in components
                    for component in matching_result.get("components", []):
                        if component.get("flag") == "critical":
                            issues.append(self.create_issue(
                                code="LAB_CRITICAL_VALUE",
                                title=f"Critical Value: {component.get('name')}",
                                severity="critical",
                                message=f"{component.get('name')} = {component.get('value')} {component.get('units')} (Reference: {component.get('reference_range')})",
                                suggested_action="Consult physician before discharge - critical lab value requires review",
                                evidence=[format_evidence_path("data/lab_results.json", f"results[{matching_result.get('test_id')}].components")],
                                data={
                                    "test": required_test,
                                    "component": component.get("name"),
                                    "value": component.get("value"),
                                    "threshold": component.get("critical_threshold")
                                }
                            ))
                            noc = False
        
        return self.create_output(
            noc=noc,
            confidence=0.8,
            issues=issues,
            raw_response={"fallback": True}
        )
