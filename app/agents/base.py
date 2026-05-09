from abc import ABC, abstractmethod
from typing import Optional, Callable, Awaitable
from app.models.schemas import SharedContext, AgentType
from app.context.shared_state import SharedContextManager
from app.logging.json_logger import get_logger
from app.context.budget_manager import ContextBudgetManager
from datetime import datetime

logger = get_logger(__name__)

class BaseAgent(ABC):
    """Base class for all agents with shared context access"""
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.context_manager = SharedContextManager()
        self.budget_manager = ContextBudgetManager()
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    async def process(
        self, 
        context: SharedContext,
        streaming_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None
    ) -> SharedContext:
        """Process the context and return updated context"""
        pass
    
    async def _log_execution(self, context: SharedContext, action: str, details: dict):
        """Log agent execution to trace"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent_type.value,
            "action": action,
            "details": details,
            "job_id": context.job_id
        }
        context.execution_trace.append(event)
        self.logger.info("agent_execution", extra={"event": event})

    async def _consume_budget(self, context: SharedContext, estimated_tokens: int, action: str) -> bool:
        """Deduct token budget for the current agent and log violations."""
        if not context.budgets:
            return True
        success = await self.budget_manager.deduct_budget(context, self.agent_type, estimated_tokens)
        if not success:
            violation = {
                "agent": self.agent_type.value,
                "action": action,
                "tokens_requested": estimated_tokens,
                "remaining": context.budgets.get(self.agent_type).remaining_tokens if self.agent_type in context.budgets else None
            }
            context.policy_violations.append(violation)
            self.logger.warning("budget_violation", extra={"violation": violation})
        return success

    def _check_budget(self, context: SharedContext, required_tokens: int) -> bool:
        """Check if agent has sufficient budget"""
        budget = context.budgets.get(self.agent_type)
        if not budget:
            return False
        return budget.remaining_tokens >= required_tokens

    async def _emit_event(self, callback, event_type: str, **payload):
        """Emit structured event with safety and async compatibility"""
        if callback is None:
            return  # Safe no-op
        
        try:
            # Build structured payload
            event_payload = {
                "timestamp": datetime.now().isoformat(),
                "agent_type": self.agent_type.value,
                "event_type": event_type,
                "trace_id": getattr(self.context_manager, 'current_context', lambda: None)().job_id if hasattr(self.context_manager, 'current_context') else None,
                **payload
            }
            
            # Safely await callback
            await callback(event_type, event_payload)
            
        except Exception as e:
            # Log failures but never crash orchestration
            import traceback
            stack_trace = traceback.format_exc()
            self.logger.error(f"Event emission failed for {event_type}: {e}", 
                            extra={"stack_trace": stack_trace, "event_payload": payload}, 
                            exc_info=True)