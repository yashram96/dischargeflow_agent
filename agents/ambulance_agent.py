from typing import Dict, Any
from agents.base_agent import BaseAgent
from schemas.agent_schema import AgentOutputSchema
from utils.file_utils import read_json_file, format_evidence_path
import google.generativeai as genai
import json
from config import Config

class AmbulanceAgent(BaseAgent):
    """
    Ambulance/Transport verification agent that assesses transport needs
    and provider availability using Gemini API.
    """
    
    def __init__(self, api_key: str = None):
        super().__init__("Ambulance")
        
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.GEMINI_MODEL)
        
    def verify(self, patient_id: str, **kwargs) -> AgentOutputSchema:
        """Verify transport requirements and availability"""
        self.start_timer()
        
        # Load data files
        patient_data = self.load_patient_data()
        transport_providers = read_json_file("data/transport_providers.json")
        
        self.add_checked_field("mobility")
        self.add_checked_field("discharge_disposition")
        self.add_checked_field("transport_providers")
        
        # Build prompt for Gemini
        prompt = self._build_verification_prompt(patient_data, transport_providers)
        
        try:
            # print("  Calling Gemini API for ambulance verification...")
            
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
                return self._fallback_verification(patient_data, transport_providers)
            
            print("  Gemini API response received, parsing...")
            result = self._parse_gemini_response(response.text)
            print(f"  ✓ Ambulance verification complete (NOC: {result['noc']})")
            
            return self.create_output(
                noc=result["noc"],
                confidence=result["confidence"],
                issues=result["issues"],
                raw_response=result.get("raw_data", {})
            )
            
        except Exception as e:
            print(f"  ✗ Gemini API error: {type(e).__name__}: {str(e)}")
            print(f"  → Falling back to rule-based verification...")
            return self._fallback_verification(patient_data, transport_providers)
    
    def _build_verification_prompt(self, patient_data: Dict, transport_providers: Dict) -> str:
        """Build verification prompt for Gemini"""
        
        # Extract relevant patient info
        patient_info = patient_data.get("Patient Information", {})
        diagnosis = patient_info.get("Current Diagnosis", "")
        conditions = patient_info.get("Existing Conditions", "")
        
        return f"""You are an Ambulance/Transport Verification Agent for hospital discharge.
Assess if patient needs ambulance transport and verify provider availability.

PATIENT INFORMATION:
- Diagnosis: {diagnosis}
- Existing Conditions: {conditions}
- Age: {patient_info.get("Age", "Unknown")}
- Address: {patient_info.get("Address", "Unknown")}

TRANSPORT PROVIDERS:
{json.dumps(transport_providers, indent=2)}

ASSESSMENT RULES:
1. Transport REQUIRED if:
   - Patient has serious condition (cancer, heart issues, dialysis)
   - Patient is elderly (>65) with multiple conditions
   - Long distance discharge (>50km)
   - Discharge to another facility
   
2. Transport type needed:
   - ICU ambulance: Critical patients, oxygen required
   - ALS ambulance: Serious conditions, monitoring needed
   - BLS ambulance: Stable patients needing medical supervision
   - Wheelchair van: Mobile patients needing assistance

3. Check provider availability and ETA (should be <120 minutes)

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
    "transport_required": true or false,
    "recommended_vehicle_type": "BLS|ALS|ICU|Wheelchair",
    "provider_available": true or false,
    "estimated_eta": 0
  }}
}}

ISSUE CODES:
- TRANSPORT_REQUIRED: Transport assessment completed, booking needed
- TRANSPORT_UNAVAILABLE: No provider available within acceptable time
- TRANSPORT_TYPE_MISMATCH: Available vehicle doesn't match patient needs

Set noc=true if transport not required OR suitable provider available with acceptable ETA.
Set noc=false if transport required but no suitable provider available.
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
    
    def _fallback_verification(self, patient_data: Dict, transport_providers: Dict) -> AgentOutputSchema:
        """Fallback rule-based verification"""
        issues = []
        noc = True
        
        # Simple rule: Liver cancer patient likely needs transport
        diagnosis = patient_data.get("Patient Information", {}).get("Current Diagnosis", "").lower()
        age = int(patient_data.get("Patient Information", {}).get("Age", "0"))
        
        transport_required = False
        if "cancer" in diagnosis or "dialysis" in str(patient_data.get("Billing", {}).get("Items", [])).lower():
            transport_required = True
        
        if transport_required:
            # Check for available providers
            if transport_providers:
                available_providers = []
                for provider in transport_providers.get("providers", []):
                    for vehicle_type, availability in provider.get("current_availability", {}).items():
                        if availability.get("available") and availability.get("eta_minutes", 999) < 120:
                            available_providers.append({
                                "provider": provider.get("name"),
                                "vehicle": vehicle_type,
                                "eta": availability.get("eta_minutes"),
                                "cost": availability.get("cost")
                            })
                
                if available_providers:
                    best_provider = min(available_providers, key=lambda x: x["eta"])
                    issues.append(self.create_issue(
                        code="TRANSPORT_REQUIRED",
                        title="Ambulance Transport Recommended",
                        severity="medium",
                        message=f"Patient with {diagnosis} should have ambulance transport arranged",
                        suggested_action=f"Book {best_provider['vehicle']} from {best_provider['provider']} (ETA: {best_provider['eta']} min, Cost: ₹{best_provider['cost']})",
                        evidence=[format_evidence_path("data/transport_providers.json", "providers")],
                        data=best_provider
                    ))
                    # noc remains True since provider is available
                else:
                    issues.append(self.create_issue(
                        code="TRANSPORT_UNAVAILABLE",
                        title="No Ambulance Available",
                        severity="high",
                        message="Transport required but no providers available within 2 hours",
                        suggested_action="Contact private ambulance services or delay discharge",
                        evidence=[format_evidence_path("data/transport_providers.json", "providers")]
                    ))
                    noc = False
        
        return self.create_output(
            noc=noc,
            confidence=0.7,
            issues=issues,
            raw_response={"transport_required": transport_required, "fallback": True}
        )
