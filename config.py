import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration management for the discharge system"""
    
    # API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # File Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Patient Data
    PATIENT_DATA_FILE = BASE_DIR / "patient_data.json"
    INSURANCE_POLICY_FILE = BASE_DIR / "insurance_policy.txt"
    
    # Mock Data Files
    LAB_RESULTS_FILE = DATA_DIR / "lab_results.json"
    PHARMACY_INVENTORY_FILE = DATA_DIR / "pharmacy_inventory.json"
    TRANSPORT_PROVIDERS_FILE = DATA_DIR / "transport_providers.json"
    BILLING_SNAPSHOT_FILE = DATA_DIR / "billing_snapshot.json"
    HOUSEKEEPING_SCHEDULE_FILE = DATA_DIR / "housekeeping_schedule.json"
    INSURER_RECORDS_FILE = DATA_DIR / "insurer_records.json"
    DRUG_INTERACTION_RULES_FILE = DATA_DIR / "drug_interaction_rules.json"
    
    # Agent Configuration
    AGENT_TIMEOUT_SECONDS = 30
    MAX_RETRIES = 2
    
    # Workflow Configuration
    APPROVAL_EXPIRY_HOURS = 6
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Please set it in .env file or environment variables."
            )
        
        # Create directories if they don't exist
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        
        return True
