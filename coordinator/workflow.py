from langgraph.graph import StateGraph, END
from typing import Dict, Any

from coordinator.workflow_state import DischargeState, create_initial_state
from agents.insurance_agent import InsuranceAgent
from agents.pharmacy_agent import PharmacyAgent
from agents.ambulance_agent import AmbulanceAgent
from agents.bed_management_agent import BedManagementAgent
from agents.lab_agent import LabAgent
from coordinator.coordinator_agent import CoordinatorAgent


class DischargeWorkflow:
    """
    LangGraph workflow for patient discharge verification.
    Orchestrates parallel execution of 5 agents and coordinator.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize workflow with all agents.
        
        Args:
            api_key: Gemini API key
        """
        self.api_key = api_key
        
        # Initialize all agents
        self.insurance_agent = InsuranceAgent(api_key)
        self.pharmacy_agent = PharmacyAgent(api_key)
        self.ambulance_agent = AmbulanceAgent(api_key)
        self.bed_agent = BedManagementAgent(api_key)
        self.lab_agent = LabAgent(api_key)
        self.coordinator = CoordinatorAgent(api_key)
        
        # Build workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create workflow graph
        workflow = StateGraph(DischargeState)
        
        # Add agent nodes
        workflow.add_node("insurance", self._run_insurance_agent)
        workflow.add_node("pharmacy", self._run_pharmacy_agent)
        workflow.add_node("ambulance", self._run_ambulance_agent)
        workflow.add_node("bed", self._run_bed_agent)
        workflow.add_node("lab", self._run_lab_agent)
        workflow.add_node("coordinator", self._run_coordinator)
        
        # Set entry point (all agents run in parallel)
        workflow.set_entry_point("insurance")
        workflow.add_edge("insurance", "pharmacy")
        workflow.add_edge("pharmacy", "ambulance")
        workflow.add_edge("ambulance", "bed")
        workflow.add_edge("bed", "lab")
        workflow.add_edge("lab", "coordinator")
        workflow.add_edge("coordinator", END)
        
        return workflow.compile()
    
    def _run_insurance_agent(self, state: DischargeState) -> Dict[str, Any]:
        """Run insurance verification agent"""
        print("🏥 Running Insurance Agent...")
        output = self.insurance_agent.verify(state["patient_id"])
        return {
            "insurance_output": self.insurance_agent.to_dict(output)
        }
    
    def _run_pharmacy_agent(self, state: DischargeState) -> Dict[str, Any]:
        """Run pharmacy verification agent"""
        print("💊 Running Pharmacy Agent...")
        output = self.pharmacy_agent.verify(state["patient_id"])
        return {
            "pharmacy_output": self.pharmacy_agent.to_dict(output)
        }
    
    def _run_ambulance_agent(self, state: DischargeState) -> Dict[str, Any]:
        """Run ambulance verification agent"""
        print("🚑 Running Ambulance Agent...")
        output = self.ambulance_agent.verify(state["patient_id"])
        return {
            "ambulance_output": self.ambulance_agent.to_dict(output)
        }
    
    def _run_bed_agent(self, state: DischargeState) -> Dict[str, Any]:
        """Run bed management verification agent"""
        print("🛏️  Running Bed Management Agent...")
        output = self.bed_agent.verify(state["patient_id"])
        return {
            "bed_output": self.bed_agent.to_dict(output)
        }
    
    def _run_lab_agent(self, state: DischargeState) -> Dict[str, Any]:
        """Run lab verification agent"""
        print("🔬 Running Lab Agent...")
        output = self.lab_agent.verify(state["patient_id"])
        return {
            "lab_output": self.lab_agent.to_dict(output)
        }
    
    def _run_coordinator(self, state: DischargeState) -> Dict[str, Any]:
        """Run coordinator to make final decision"""
        print("🎯 Running Coordinator Agent...")
        
        # Gather all agent outputs
        agent_outputs = {
            "Insurance": state["insurance_output"],
            "Pharmacy": state["pharmacy_output"],
            "Ambulance": state["ambulance_output"],
            "Bed Management": state["bed_output"],
            "Lab": state["lab_output"]
        }
        
        # Coordinate and make decision
        decision = self.coordinator.coordinate(state["patient_id"], agent_outputs)
        
        return {
            "all_agents_complete": True,
            "final_decision": decision.final_decision,
            "approved": decision.approved,
            "approved_by": decision.approved_by,
            "blocked_by": decision.blocked_by,
            "aggregated_issues": decision.issues,
            "discharge_summary": decision.discharge_summary,
            "suggested_auto_resolutions": decision.suggested_auto_resolutions,
            "timestamp": decision.timestamp,
            "files_written": decision.files_written
        }
    
    def run(self, patient_id: str) -> DischargeState:
        """
        Execute the discharge workflow for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Final workflow state
        """
        print(f"\n{'='*60}")
        print(f"🏥 PATIENT DISCHARGE VERIFICATION WORKFLOW")
        print(f"{'='*60}")
        print(f"Patient ID: {patient_id}\n")
        
        # Create initial state
        initial_state = create_initial_state(patient_id)
        
        # Run workflow
        final_state = self.workflow.invoke(initial_state)
        
        print(f"\n{'='*60}")
        print(f"✅ WORKFLOW COMPLETE")
        print(f"{'='*60}")
        print(f"Final Decision: {final_state['final_decision']}")
        print(f"Approved: {final_state['approved']}")
        print(f"Approved By: {', '.join(final_state['approved_by'])}")
        if final_state['blocked_by']:
            print(f"Blocked By: {', '.join(final_state['blocked_by'])}")
        print(f"\nFiles Written:")
        for file in final_state['files_written']:
            print(f"  - {file}")
        print(f"{'='*60}\n")
        
        return final_state
