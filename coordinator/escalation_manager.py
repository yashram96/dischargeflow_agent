from typing import Dict, Any, List
from datetime import datetime
import json
import os
from pathlib import Path

from schemas.agent_schema import EscalationAlert
from utils.file_utils import get_iso_timestamp, write_json_file


class EscalationManager:
    """
    Manages escalation alerts for different departments based on discharge issues.
    Generates department-specific alert files that can be consumed by respective portals.
    """
    
    # Mapping of issue code prefixes to departments
    DEPARTMENT_MAPPING = {
        "LAB_": "Lab Portal",
        "PHARM_": "Pharmacy Portal",
        "BED_": "Billing Portal",
        "BILLING_": "Billing Portal",
        "TRANSPORT_": "Transport Services",
        "INS_": "Insurance Desk"
    }
    
    # Severity to priority mapping
    SEVERITY_TO_PRIORITY = {
        "critical": "urgent",
        "high": "high",
        "medium": "normal",
        "low": "low"
    }
    
    def __init__(self, escalations_dir: str = "escalations"):
        """
        Initialize escalation manager.
        
        Args:
            escalations_dir: Base directory for escalation files
        """
        self.escalations_dir = escalations_dir
        self.alert_counter = 0
    
    def create_escalations(
        self,
        patient_id: str,
        issues: List[Dict[str, Any]],
        final_decision: str
    ) -> List[str]:
        """
        Create escalation alerts for all issues.
        
        Args:
            patient_id: Patient identifier
            issues: List of issues from all agents
            final_decision: Final discharge decision
            
        Returns:
            List of file paths written
        """
        if not issues:
            return []
        
        # Create patient-specific directory
        patient_dir = os.path.join(self.escalations_dir, f"patient_{patient_id}")
        os.makedirs(patient_dir, exist_ok=True)
        
        # Group issues by department
        department_alerts = {}
        patient_notifications = []
        
        for issue in issues:
            # Create alert
            alert = self._create_alert(patient_id, issue)
            
            # Add to department group
            dept = alert.department
            if dept not in department_alerts:
                department_alerts[dept] = []
            department_alerts[dept].append(alert)
            
            # Add to patient notifications if critical/high
            if alert.priority in ["urgent", "high"]:
                patient_notifications.append(alert)
        
        # Write department-specific alert files
        files_written = []
        
        for department, alerts in department_alerts.items():
            file_path = self._write_department_alerts(patient_dir, department, alerts)
            files_written.append(file_path)
        
        # Write patient notifications
        if patient_notifications:
            file_path = self._write_patient_notifications(patient_dir, patient_notifications)
            files_written.append(file_path)
        
        # Write escalation summary
        summary_path = self._write_escalation_summary(
            patient_dir, patient_id, department_alerts, final_decision
        )
        files_written.append(summary_path)
        
        return files_written
    
    def _create_alert(self, patient_id: str, issue: Dict[str, Any]) -> EscalationAlert:
        """Create an escalation alert from an issue."""
        self.alert_counter += 1
        
        issue_code = issue.get("code", "UNKNOWN")
        department = self._map_issue_to_department(issue_code)
        severity = issue.get("severity", "medium")
        priority = self.SEVERITY_TO_PRIORITY.get(severity, "normal")
        
        # Generate unique alert ID
        dept_abbrev = department.replace(" ", "").replace("Portal", "").upper()[:3]
        alert_id = f"ALERT-{patient_id}-{dept_abbrev}-{self.alert_counter:03d}"
        
        return EscalationAlert(
            alert_id=alert_id,
            patient_id=patient_id,
            department=department,
            priority=priority,
            issue_code=issue_code,
            issue_title=issue.get("title", ""),
            message=issue.get("message", ""),
            suggested_action=issue.get("suggested_action", ""),
            evidence=issue.get("evidence", []),
            data=issue.get("data", {}),
            escalated_at=get_iso_timestamp(),
            status="pending"
        )
    
    def _map_issue_to_department(self, issue_code: str) -> str:
        """Map issue code to department."""
        for prefix, department in self.DEPARTMENT_MAPPING.items():
            if issue_code.startswith(prefix):
                return department
        return "General Operations"
    
    def _write_department_alerts(
        self,
        patient_dir: str,
        department: str,
        alerts: List[EscalationAlert]
    ) -> str:
        """Write department-specific alert file."""
        # Create filename from department name
        filename = department.lower().replace(" ", "_") + ".json"
        file_path = os.path.join(patient_dir, filename)
        
        # Get highest priority
        priority_order = ["urgent", "high", "normal", "low"]
        highest_priority = min(
            [alert.priority for alert in alerts],
            key=lambda p: priority_order.index(p)
        )
        
        # Create department alert structure
        department_data = {
            "department": department,
            "patient_id": alerts[0].patient_id,
            "alerts": [alert.model_dump() for alert in alerts],
            "total_alerts": len(alerts),
            "highest_priority": highest_priority,
            "generated_at": get_iso_timestamp()
        }
        
        write_json_file(file_path, department_data)
        return file_path
    
    def _write_patient_notifications(
        self,
        patient_dir: str,
        alerts: List[EscalationAlert]
    ) -> str:
        """Write patient notification file."""
        file_path = os.path.join(patient_dir, "patient_notifications.json")
        
        # Create patient-friendly notifications
        notifications = []
        for alert in alerts:
            notifications.append({
                "priority": alert.priority,
                "title": alert.issue_title,
                "message": self._create_patient_message(alert),
                "action_required": alert.suggested_action,
                "department": alert.department
            })
        
        notification_data = {
            "patient_id": alerts[0].patient_id,
            "notifications": notifications,
            "total_notifications": len(notifications),
            "generated_at": get_iso_timestamp()
        }
        
        write_json_file(file_path, notification_data)
        return file_path
    
    def _create_patient_message(self, alert: EscalationAlert) -> str:
        """Create patient-friendly message from alert."""
        # Simplify technical messages for patients
        if "Lab" in alert.department:
            return f"Your {alert.issue_title.lower()} needs attention. Please contact the lab."
        elif "Billing" in alert.department:
            return f"There is a billing matter that needs your attention: {alert.message}"
        elif "Pharmacy" in alert.department:
            return f"Your medication {alert.issue_title.lower()} requires action."
        elif "Insurance" in alert.department:
            return f"Please contact the insurance desk regarding: {alert.issue_title}"
        else:
            return alert.message
    
    def _write_escalation_summary(
        self,
        patient_dir: str,
        patient_id: str,
        department_alerts: Dict[str, List[EscalationAlert]],
        final_decision: str
    ) -> str:
        """Write escalation summary file."""
        file_path = os.path.join(patient_dir, f"escalation_summary_{patient_id}.json")
        
        # Count alerts by priority
        priority_counts = {"urgent": 0, "high": 0, "normal": 0, "low": 0}
        total_alerts = 0
        
        for alerts in department_alerts.values():
            for alert in alerts:
                priority_counts[alert.priority] += 1
                total_alerts += 1
        
        summary_data = {
            "patient_id": patient_id,
            "final_decision": final_decision,
            "total_alerts": total_alerts,
            "alerts_by_priority": priority_counts,
            "departments_involved": list(department_alerts.keys()),
            "department_summary": {
                dept: len(alerts) for dept, alerts in department_alerts.items()
            },
            "generated_at": get_iso_timestamp()
        }
        
        write_json_file(file_path, summary_data)
        return file_path
