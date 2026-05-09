"""
Base Tool Class - All tools inherit from this
Defines failure contracts and logging interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class ToolResult:
    """Standard result format for all tools"""
    success: bool
    data: Any
    error: Optional[str] = None
    latency_ms: float = 0
    input_hash: Optional[str] = None
    output_hash: Optional[str] = None
    retry_count: int = 0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class BaseTool(ABC):
    """Base class for all tools with failure contracts"""
    
    def __init__(self, tool_name: str, max_retries: int = 2):
        self.tool_name = tool_name
        self.max_retries = max_retries
        self.call_history = []
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool. Must handle all failure modes."""
        pass
    
    async def call_with_retry(self, **kwargs) -> ToolResult:
        """Call tool with retry logic on failure"""
        for attempt in range(self.max_retries + 1):
            try:
                result = await self.execute(**kwargs)
                result.retry_count = attempt
                self.call_history.append({
                    "attempt": attempt,
                    "input": kwargs,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                })
                if result.success or attempt == self.max_retries:
                    return result
            except Exception as e:
                logger.error(f"Tool {self.tool_name} attempt {attempt} failed: {e}")
                if attempt == self.max_retries:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Max retries exceeded: {str(e)}",
                        retry_count=attempt
                    )
        return ToolResult(success=False, data=None, error="Unknown error")
    
    def get_history(self) -> list:
        """Get call history for audit logging"""
        return self.call_history
    
    def _handle_timeout(self, timeout_seconds: int) -> ToolResult:
        """Default timeout handler"""
        return ToolResult(
            success=False,
            data=None,
            error=f"Tool timeout after {timeout_seconds}s",
            latency_ms=timeout_seconds * 1000
        )
    
    def _handle_empty_results(self) -> ToolResult:
        """Default empty results handler"""
        return ToolResult(
            success=True,
            data=[],
            error=None
        )
    
    def _handle_malformed_input(self, error: str) -> ToolResult:
        """Default malformed input handler"""
        return ToolResult(
            success=False,
            data=None,
            error=f"Malformed input: {error}"
        )
