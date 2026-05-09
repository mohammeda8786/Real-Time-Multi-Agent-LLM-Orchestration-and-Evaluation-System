from abc import ABC, abstractmethod
from typing import Optional, Callable, Awaitable
from app.models.schemas import SharedContext, AgentType
from app.context.shared_state import SharedContextManager
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents with shared context access"""
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.context_manager = SharedContextManager()
    
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
        context.execution_trace.append({
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent_type.value,
            "action": action,
            "details": details,
            "job_id": context.job_id
        })
    
    async def _emit_event(self, callback, event_type: str, **kwargs):
        """Emit SSE event if callback provided"""
        if callback:
            await callback(event_type, kwargs)
    
    def _check_budget(self, context: SharedContext, required_tokens: int) -> bool:
        """Check if agent has sufficient budget"""
        budget = context.budgets.get(self.agent_type)
        if not budget:
            return False
        return budget.remaining_tokens >= required_tokens