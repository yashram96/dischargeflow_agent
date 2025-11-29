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


def print_escalations(patient_id):
    """Print escalation alerts in a beautiful format"""
    import os
    from pathlib import Path
    
    escalations_dir = Path("escalations") / f"patient_{patient_id}"
    
    if not escalations_dir.exists():
        return
    
    print("\n" + "="*70)
    print("🚨 DEPARTMENT ESCALATION ALERTS")
    print("="*70)
    
    # Priority icons
    priority_icons = {
        "urgent": "🔴",
        "high": "🟠",
        "normal": "🟡",
        "low": "🟢"
    }
    
    # Department icons
    dept_icons = {
        "Lab Portal": "🔬",
        "Pharmacy Portal": "💊",
        "Billing Portal": "💰",
        "Transport Services": "🚑",
        "Insurance Desk": "📋",
        "General Operations": "⚙️"
    }
    
    # Read and display each department alert file
    alert_files = sorted(escalations_dir.glob("*.json"))
    
    for alert_file in alert_files:
        if alert_file.name == f"escalation_summary_{patient_id}.json":
            continue  # Skip summary, we'll show it at the end
        
        if alert_file.name == "patient_notifications.json":
            continue  # Skip patient notifications for now
        
        try:
            with open(alert_file, 'r', encoding='utf-8') as f:
                dept_data = json.load(f)
            
            department = dept_data.get("department", "Unknown")
            alerts = dept_data.get("alerts", [])
            total = dept_data.get("total_alerts", 0)
            highest_priority = dept_data.get("highest_priority", "normal")
            
            if not alerts:
                continue
            
            # Print department header
            dept_icon = dept_icons.get(department, "📌")
            priority_icon = priority_icons.get(highest_priority, "⚪")
            
            print(f"\n{dept_icon} {department.upper()} {priority_icon}")
            print("─" * 70)
            print(f"Total Alerts: {total} | Highest Priority: {highest_priority.upper()}")
            print()
            
            # Print each alert
            for i, alert in enumerate(alerts, 1):
                priority = alert.get("priority", "normal")
                title = alert.get("issue_title", "")
                message = alert.get("message", "")
                action = alert.get("suggested_action", "")
                alert_id = alert.get("alert_id", "")
                
                print(f"  {priority_icons.get(priority, '⚪')} Alert #{i}: {title}")
                print(f"     ID: {alert_id}")
                print(f"     Priority: {priority.upper()}")
                
                # Wrap message to fit terminal width
                if message:
                    import textwrap
                    wrapped_msg = textwrap.fill(message, width=65, initial_indent="     ", subsequent_indent="     ")
                    print(f"     Issue: {message[:100]}{'...' if len(message) > 100 else ''}")
                
                if action:
                    print(f"     ✓ Action: {action[:80]}{'...' if len(action) > 80 else ''}")
                
                print()
        
        except Exception as e:
            print(f"  ⚠️  Error reading {alert_file.name}: {e}")
    
    # Print patient notifications if they exist
    patient_notif_file = escalations_dir / "patient_notifications.json"
    if patient_notif_file.exists():
        try:
            with open(patient_notif_file, 'r', encoding='utf-8') as f:
                notif_data = json.load(f)
            
            notifications = notif_data.get("notifications", [])
            
            if notifications:
                print(f"\n👤 PATIENT/FAMILY NOTIFICATIONS")
                print("─" * 70)
                print(f"Total Notifications: {len(notifications)}")
                print()
                
                for i, notif in enumerate(notifications, 1):
                    priority = notif.get("priority", "normal")
                    title = notif.get("title", "")
                    message = notif.get("message", "")
                    
                    print(f"  {priority_icons.get(priority, '⚪')} Notification #{i}: {title}")
                    print(f"     {message}")
                    print()
        
        except Exception as e:
            print(f"  ⚠️  Error reading patient notifications: {e}")
    
    # Print escalation summary
    summary_file = escalations_dir / f"escalation_summary_{patient_id}.json"
    if summary_file.exists():
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            print(f"\n📊 ESCALATION SUMMARY")
            print("─" * 70)
            print(f"Total Alerts: {summary.get('total_alerts', 0)}")
            print(f"Departments Involved: {', '.join(summary.get('departments_involved', []))}")
            
            alerts_by_priority = summary.get('alerts_by_priority', {})
            if alerts_by_priority:
                print(f"\nAlerts by Priority:")
                for priority in ["urgent", "high", "normal", "low"]:
                    count = alerts_by_priority.get(priority, 0)
                    if count > 0:
                        print(f"  {priority_icons.get(priority, '⚪')} {priority.upper()}: {count}")
            
            dept_summary = summary.get('department_summary', {})
            if dept_summary:
                print(f"\nAlerts by Department:")
                for dept, count in dept_summary.items():
                    dept_icon = dept_icons.get(dept, "📌")
                    print(f"  {dept_icon} {dept}: {count}")
        
        except Exception as e:
            print(f"  ⚠️  Error reading escalation summary: {e}")
    
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
        
        # Print escalation alerts
        print_escalations(patient_id)
        
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
