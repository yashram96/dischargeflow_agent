#!/usr/bin/env python3
"""
Simple test script to verify individual agents work correctly.
"""

import sys
from config import Config

# Validate config
try:
    Config.validate()
    print("✅ Configuration validated")
    print(f"API Key present: {bool(Config.GEMINI_API_KEY)}")
    print(f"API Key length: {len(Config.GEMINI_API_KEY)}")
except Exception as e:
    print(f"❌ Config error: {e}")
    sys.exit(1)

# Test Insurance Agent
print("\n" + "="*60)
print("Testing Insurance Agent")
print("="*60)

try:
    from agents.insurance_agent import InsuranceAgent
    
    print("Creating Insurance Agent...")
    agent = InsuranceAgent(api_key=Config.GEMINI_API_KEY)
    
    print("Running verification for patient P00231...")
    result = agent.verify("P00231")
    
    print("\n✅ Insurance Agent completed!")
    print(f"NOC: {result.noc}")
    print(f"Confidence: {result.confidence}")
    print(f"Issues: {len(result.issues)}")
    
    if result.issues:
        print("\nIssues found:")
        for issue in result.issues:
            print(f"  - [{issue.severity}] {issue.title}: {issue.message}")
    
    print(f"\nExecution time: {result.meta.time_ms:.2f}ms")
    
except Exception as e:
    print(f"\n❌ Insurance Agent failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ Test completed successfully!")
print("="*60)
