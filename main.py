#!/usr/bin/env python3
"""
Patient Discharge Automation System
Main entry point for the multi-agent discharge verification workflow.
"""

import sys
import json
from pathlib import Path

from config import Config
from coordinator.workflow import DischargeWorkflow


def print_banner():
    """Print application banner"""
    print("\n" + "="*70)
    print("🏥 PATIENT DISCHARGE AUTOMATION SYSTEM")
    print("="*70)
    print("Multi-Agent Verification using LangGraph & Gemini AI")
    print("="*70 + "\n")


def print_decision_summary(final_state):
    """Print formatted decision summary"""
    print("\n" + "="*70)
    print("📋 DISCHARGE DECISION SUMMARY")
    print("="*70)
    
    print(f"\n🆔 Patient ID: {final_state['patient_id']}")
    print(f"⏰ Timestamp: {final_state['timestamp']}")
    
    print(f"\n🎯 Final Decision: {final_state['final_decision']}")
    print(f"✅ Approved: {'YES' if final_state['approved'] else 'NO'}")
    
    if final_state['approved_by']:
        print(f"\n✓ Approved By:")
        for agent in final_state['approved_by']:
            print(f"  • {agent}")
    
    if final_state['blocked_by']:
        print(f"\n✗ Blocked By:")
        for agent in final_state['blocked_by']:
            print(f"  • {agent}")
    
    # Print issues
    if final_state['aggregated_issues']:
        print(f"\n⚠️  Issues Found ({len(final_state['aggregated_issues'])}):")
        
        # Group by severity
        critical = [i for i in final_state['aggregated_issues'] if i.get('severity') == 'critical']
        high = [i for i in final_state['aggregated_issues'] if i.get('severity') == 'high']
        medium = [i for i in final_state['aggregated_issues'] if i.get('severity') == 'medium']
        low = [i for i in final_state['aggregated_issues'] if i.get('severity') == 'low']
        
        if critical:
            print(f"\n  🔴 CRITICAL ({len(critical)}):")
            for issue in critical:
                print(f"    • [{issue.get('agent')}] {issue.get('title')}")
                print(f"      {issue.get('message')}")
        
        if high:
            print(f"\n  🟠 HIGH ({len(high)}):")
            for issue in high:
                print(f"    • [{issue.get('agent')}] {issue.get('title')}")
                print(f"      {issue.get('message')}")
        
        if medium:
            print(f"\n  🟡 MEDIUM ({len(medium)}):")
            for issue in medium:
                print(f"    • [{issue.get('agent')}] {issue.get('title')}")
        
        if low:
            print(f"\n  🟢 LOW ({len(low)}):")
            for issue in low:
                print(f"    • [{issue.get('agent')}] {issue.get('title')}")
    
    # Print discharge summary
    if final_state.get('discharge_summary'):
        print(f"\n📄 Discharge Summary:")
        print(f"\n  Patient/Family Summary:")
        print(f"  {final_state['discharge_summary'].get('plain_text', 'N/A')}")
        
        print(f"\n  Medical Record Summary:")
        print(f"  {final_state['discharge_summary'].get('for_medical_record', 'N/A')}")
    
    # Print auto-resolutions
    if final_state.get('suggested_auto_resolutions'):
        print(f"\n🔧 Suggested Auto-Resolutions ({len(final_state['suggested_auto_resolutions'])}):")
        for i, resolution in enumerate(final_state['suggested_auto_resolutions'], 1):
            print(f"  {i}. {resolution.get('action')}")
    
    # Print output files
    if final_state.get('files_written'):
        print(f"\n💾 Output Files:")
        for file in final_state['files_written']:
            print(f"  • {file}")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main application entry point"""
    print_banner()
    
    # Validate configuration
    try:
        Config.validate()
        print("✅ Configuration validated")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\nPlease create a .env file with:")
        print("GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Get patient ID (default to P00231 from patient_data.json)
    patient_id = sys.argv[1] if len(sys.argv) > 1 else "P00231"
    
    print(f"🔍 Initiating discharge verification for Patient ID: {patient_id}\n")
    
    try:
        # Initialize workflow
        print("🚀 Initializing discharge workflow...")
        workflow = DischargeWorkflow(api_key=Config.GEMINI_API_KEY)
        
        # Run workflow
        final_state = workflow.run(patient_id)
        
        # Print summary
        print_decision_summary(final_state)
        
        # Save final state to JSON for easy viewing
        output_file = Path("output") / f"final_decision_{patient_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            # Convert to serializable format
            serializable_state = {
                k: v for k, v in final_state.items()
                if k not in ['__pydantic_extra__', '__pydantic_fields_set__']
            }
            json.dump(serializable_state, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Full decision saved to: {output_file}")
        
        # Exit code based on decision
        if final_state['approved']:
            print("\n✅ Discharge APPROVED - Patient ready for discharge")
            sys.exit(0)
        else:
            print(f"\n❌ Discharge {final_state['final_decision']} - Action required")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error during workflow execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
