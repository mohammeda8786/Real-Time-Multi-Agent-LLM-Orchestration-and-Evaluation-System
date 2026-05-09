"""
Context Budget Manager - Tracks token consumption and enforces budget limits
"""

from app.models.schemas import SharedContext, AgentType, ContextBudget
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class ContextBudgetManager:
    """Manages context window budgets across all agents"""
    
    def __init__(self, total_context_window: int = 8000):
        self.total_context_window = total_context_window
        self.per_agent_budgets = {
            AgentType.DECOMPOSER: 1000,
            AgentType.RAG: 2500,
            AgentType.CRITIC: 1500,
            AgentType.SYNTHESIZER: 1500,
            AgentType.ORCHESTRATOR: 500
        }
    
    async def initialize_budgets(self, context: SharedContext):
        """Initialize budget allocations for all agents"""
        context.budgets = {}
        
        for agent_type, allocated in self.per_agent_budgets.items():
            context.budgets[agent_type] = ContextBudget(
                agent_type=agent_type,
                allocated_tokens=allocated,
                used_tokens=0,
                remaining_tokens=allocated
            )
        
        logger.info(f"Initialized budgets for job {context.job_id}")
    
    async def check_budget(self, context: SharedContext, agent_type: AgentType, required_tokens: int) -> tuple[bool, str]:
        """Check if agent has sufficient budget"""
        if agent_type not in context.budgets:
            return False, f"Agent {agent_type} not found in budgets"
        
        budget = context.budgets[agent_type]
        
        if budget.remaining_tokens < required_tokens:
            return False, (
                f"Agent {agent_type} insufficient budget. "
                f"Required: {required_tokens}, Available: {budget.remaining_tokens}"
            )
        
        return True, "Budget available"
    
    async def deduct_budget(self, context: SharedContext, agent_type: AgentType, tokens_used: int) -> bool:
        """Deduct tokens from agent budget"""
        if agent_type not in context.budgets:
            logger.error(f"Agent {agent_type} not found in budgets")
            return False
        
        budget = context.budgets[agent_type]
        
        if budget.remaining_tokens < tokens_used:
            violation = {
                "type": "budget_overflow",
                "agent": agent_type,
                "requested": tokens_used,
                "available": budget.remaining_tokens,
                "severity": "high"
            }
            context.policy_violations.append(violation)
            logger.warning(f"Budget overflow detected: {violation}")
            return False
        
        budget.used_tokens += tokens_used
        budget.remaining_tokens -= tokens_used
        
        logger.info(f"Agent {agent_type} deducted {tokens_used} tokens, remaining: {budget.remaining_tokens}")
        return True
    
    async def get_remaining_budget(self, context: SharedContext, agent_type: AgentType) -> int:
        """Get remaining budget for an agent"""
        if agent_type not in context.budgets:
            return 0
        return context.budgets[agent_type].remaining_tokens
    
    def get_budget_report(self, context: SharedContext) -> Dict:
        """Get comprehensive budget report"""
        total_allocated = sum(b.allocated_tokens for b in context.budgets.values())
        total_used = sum(b.used_tokens for b in context.budgets.values())
        total_remaining = sum(b.remaining_tokens for b in context.budgets.values())
        
        return {
            "total_context_window": self.total_context_window,
            "total_allocated": total_allocated,
            "total_used": total_used,
            "total_remaining": total_remaining,
            "per_agent": {
                agent_type: {
                    "allocated": b.allocated_tokens,
                    "used": b.used_tokens,
                    "remaining": b.remaining_tokens,
                    "utilization_percent": (b.used_tokens / b.allocated_tokens * 100) if b.allocated_tokens > 0 else 0
                }
                for agent_type, b in context.budgets.items()
            }
        }