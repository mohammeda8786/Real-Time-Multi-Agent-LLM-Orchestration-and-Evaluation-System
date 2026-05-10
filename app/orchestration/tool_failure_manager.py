"""
Centralized policy manager for tool retries, backoff, and safe fallback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional

from app.models.schemas import SharedContext
from app.tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolFailurePolicyManager:
    def __init__(self, max_retries: int = 2, base_backoff: float = 0.5, max_backoff: float = 3.0):
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff

    async def execute_tool(
        self,
        tool: BaseTool,
        tool_name: str,
        context: SharedContext,
        args: Dict[str, Any],
        streaming_callback: Optional[Callable[[str, dict], Any]] = None,
    ) -> ToolResult:
        if streaming_callback:
            await streaming_callback(
                "tool_execution_start",
                {
                    "tool_name": tool_name,
                    "args_preview": str(args)[:300],
                    "job_id": context.job_id,
                },
            )

        start_time = time.monotonic()
        result: ToolResult = ToolResult(success=False, data=None, error="tool execution not attempted")
        for attempt in range(self.max_retries + 1):
            try:
                result = await tool.execute(**args)
                result.retry_count = attempt
                if result.success:
                    break
                if attempt == self.max_retries:
                    break
            except Exception as exc:
                result = ToolResult(success=False, data=None, error=str(exc), retry_count=attempt)
                logger.warning(
                    "Tool %s failed on attempt %s: %s",
                    tool_name,
                    attempt,
                    exc,
                    exc_info=True,
                )
            if attempt < self.max_retries:
                backoff = min(self.max_backoff, self.base_backoff * (2 ** attempt))
                await asyncio.sleep(backoff)

        latency_ms = (time.monotonic() - start_time) * 1000
        if streaming_callback:
            await streaming_callback(
                "tool_execution_end",
                {
                    "tool_name": tool_name,
                    "success": result.success,
                    "error": result.error,
                    "retry_count": result.retry_count,
                    "latency_ms": round(latency_ms, 2),
                    "job_id": context.job_id,
                },
            )

        if not result.success:
            fallback = self._fallback_result(tool_name, args, context)
            if fallback:
                logger.warning(
                    "Tool %s exceeded retries; applying fallback result", tool_name
                )
                return fallback

        return result

    def _fallback_result(self, tool_name: str, args: Dict[str, Any], context: SharedContext) -> Optional[ToolResult]:
        if tool_name == "web_search":
            return ToolResult(success=True, data=[], error="fallback_empty_web_search")
        if tool_name == "sql_lookup":
            return ToolResult(success=True, data=[], error="fallback_empty_sql_lookup")
        if tool_name == "python_sandbox":
            return ToolResult(
                success=True,
                data={"output": "", "error": "sandbox unavailable; fallback safe response applied"},
                error="fallback_python_sandbox",
            )
        if tool_name == "self_reflection":
            return ToolResult(success=True, data={}, error="fallback_self_reflection")
        return None
