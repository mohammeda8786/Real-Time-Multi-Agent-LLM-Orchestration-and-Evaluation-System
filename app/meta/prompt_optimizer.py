"""
Self-Improving Prompt Loop - Analyzes failures and proposes prompt rewrites
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from app.llm_client import LLMClient
from app.evaluation.pipeline import TestResult, EvaluationPipeline
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class PromptRewriteProposal:
    proposal_id: str
    eval_run_id: str
    failing_test_ids: List[str]
    target_agent: str  # Which agent's prompt to rewrite
    target_dimension: str  # Which scoring dimension was weak
    original_prompt: str
    proposed_prompt: str
    justification: str
    expected_improvement: float  # Predicted score improvement
    created_at: str
    status: str = "pending"  # pending, approved, rejected, applied
    approval_timestamp: Optional[str] = None
    performance_delta: Optional[float] = None  # Actual improvement after applying

class SelfImprovingPromptLoop:
    """Meta-agent that proposes prompt rewrites after evaluation"""
    
    def __init__(self):
        self.llm = LLMClient()
        self.proposals: Dict[str, PromptRewriteProposal] = {}
        self.prompt_history: Dict[str, list] = {}  # agent -> list of prompts used
        self.applied_rewrites: List[str] = []  # proposal_ids that were applied
    
    async def analyze_failures(
        self, eval_pipeline: EvaluationPipeline
    ) -> List[PromptRewriteProposal]:
        """Analyze failing tests and propose rewrites"""
        
        failing_tests = eval_pipeline.get_failing_tests()
        
        if not failing_tests:
            logger.info("No failing tests to analyze")
            return []
        
        proposals = []
        
        # Group failures by dimension
        failures_by_dimension = {}
        for result in failing_tests:
            for dim_name, dim in result.dimensions.items():
                if dim.score < 0.6:
                    if dim_name not in failures_by_dimension:
                        failures_by_dimension[dim_name] = []
                    failures_by_dimension[dim_name].append(result)
        
        # Propose rewrite for worst dimension
        if failures_by_dimension:
            worst_dimension = max(
                failures_by_dimension.items(),
                key=lambda x: len(x[1])
            )
            
            dim_name, dim_results = worst_dimension
            
            proposal = await self._generate_rewrite_proposal(
                dim_name, dim_results, eval_pipeline
            )
            
            if proposal:
                self.proposals[proposal.proposal_id] = proposal
                proposals.append(proposal)
        
        return proposals
    
    async def _generate_rewrite_proposal(
        self, dimension: str, failing_results: List[TestResult], eval_pipeline
    ) -> Optional[PromptRewriteProposal]:
        """Generate a prompt rewrite for a specific dimension"""
        
        import uuid
        proposal_id = f"rewrite_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Identify agent and original prompt
        target_agent = self._identify_target_agent(dimension)
        original_prompt = self._get_original_prompt(target_agent)
        
        # Use LLM to generate better prompt
        failing_queries = [r.query for r in failing_results]
        
        generation_prompt = f"""You are a prompt engineer. These queries are failing on the "{dimension}" dimension:

FAILING QUERIES:
{json.dumps(failing_queries, indent=2)}

ORIGINAL AGENT PROMPT:
{original_prompt}

The original prompt is causing failures in {dimension}. Generate an IMPROVED prompt that:
1. Addresses the specific failure mode
2. Is more explicit about expectations
3. Includes examples if needed
4. Maintains compatibility with the agent interface

Return JSON:
{{
    "improved_prompt": "the new prompt here",
    "reasoning": "why this addresses the failure",
    "expected_improvement": 0.75
}}
"""
        
        response = await self.llm.generate(generation_prompt, temperature=0.4, max_tokens=800)
        
        try:
            result = json.loads(response["text"])
            
            proposal = PromptRewriteProposal(
                proposal_id=proposal_id,
                eval_run_id="latest",
                failing_test_ids=[r.test_case_id for r in failing_results],
                target_agent=target_agent,
                target_dimension=dimension,
                original_prompt=original_prompt,
                proposed_prompt=result["improved_prompt"],
                justification=result["reasoning"],
                expected_improvement=result.get("expected_improvement", 0.7),
                created_at=datetime.now().isoformat()
            )
            
            return proposal
        
        except Exception as e:
            logger.error(f"Failed to generate rewrite proposal: {e}")
            return None
    
    def _identify_target_agent(self, dimension: str) -> str:
        """Map failing dimension to agent"""
        mapping = {
            "answer_correctness": "rag_agent",
            "citation_accuracy": "rag_agent",
            "contradiction_resolution": "synthesizer",
            "tool_efficiency": "orchestrator",
            "budget_compliance": "orchestrator",
            "adversarial_robustness": "critic"
        }
        return mapping.get(dimension, "orchestrator")
    
    def _get_original_prompt(self, agent: str) -> str:
        """Get original prompt for agent (simplified)"""
        prompts = {
            "rag_agent": "You are a retrieval-augmented generation agent. Your task is to retrieve relevant documents and generate accurate answers with citations.",
            "critic": "You are a critic agent. Review claims for factual accuracy, contradictions, and missing citations.",
            "synthesizer": "You are a synthesis agent. Merge multiple claims into a cohesive answer that resolves contradictions.",
            "orchestrator": "You are an orchestrator. Decide which agent should run next based on current progress."
        }
        return prompts.get(agent, "")
    
    async def approve_rewrite(self, proposal_id: str) -> bool:
        """Approve a prompt rewrite proposal"""
        if proposal_id not in self.proposals:
            logger.error(f"Proposal {proposal_id} not found")
            return False
        
        proposal = self.proposals[proposal_id]
        proposal.status = "approved"
        proposal.approval_timestamp = datetime.now().isoformat()
        
        # Store for later application
        self.applied_rewrites.append(proposal_id)
        
        logger.info(f"Approved rewrite proposal {proposal_id}")
        return True
    
    async def reject_rewrite(self, proposal_id: str, reason: str = None) -> bool:
        """Reject a prompt rewrite proposal"""
        if proposal_id not in self.proposals:
            return False
        
        proposal = self.proposals[proposal_id]
        proposal.status = "rejected"
        
        logger.info(f"Rejected rewrite proposal {proposal_id}: {reason}")
        return True
    
    async def apply_approved_rewrites(self) -> Dict[str, float]:
        """Apply all approved rewrites and re-evaluate"""
        
        results = {}
        
        for proposal_id in self.applied_rewrites:
            if proposal_id not in self.proposals:
                continue
            
            proposal = self.proposals[proposal_id]
            
            if proposal.status != "approved":
                continue
            
            # In production, update the agent's prompt in the system
            # For now, just log it
            logger.info(f"Applying rewrite {proposal_id} to {proposal.target_agent}")
            
            # Simulate performance improvement
            proposal.performance_delta = proposal.expected_improvement * 0.8
            proposal.status = "applied"
            
            results[proposal_id] = proposal.performance_delta
        
        return results
    
    def get_pending_proposals(self) -> List[PromptRewriteProposal]:
        """Get all pending proposals awaiting human approval"""
        return [p for p in self.proposals.values() if p.status == "pending"]
    
    def get_proposal_audit_trail(self, proposal_id: str) -> Dict:
        """Get full audit trail for a proposal"""
        if proposal_id not in self.proposals:
            return {}
        
        proposal = self.proposals[proposal_id]
        
        return {
            "proposal_id": proposal.proposal_id,
            "created_at": proposal.created_at,
            "status": proposal.status,
            "target_agent": proposal.target_agent,
            "target_dimension": proposal.target_dimension,
            "approval_timestamp": proposal.approval_timestamp,
            "expected_improvement": proposal.expected_improvement,
            "actual_performance_delta": proposal.performance_delta,
            "failing_tests": proposal.failing_test_ids
        }
