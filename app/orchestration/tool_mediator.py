"""
Orchestrator-mediated tool execution (agents must not import tools directly).
Uses contextvars so SharedContext stays JSON-serializable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.models.schemas import SharedContext
from app.tools.web_search import WebSearchTool
from app.tools.code_execution import CodeExecutionTool
from app.tools.sql_lookup import SQLLookupTool
from app.tools.self_reflection import SelfReflectionTool

logger = logging.getLogger(__name__)

_tool_mediator_ctx: ContextVar[Optional["ToolMediator"]] = ContextVar("tool_mediator", default=None)


def get_mediator() -> Optional["ToolMediator"]:
    return _tool_mediator_ctx.get()


@dataclass
class ToolMediator:
    """Budgeted, deduplicated tool calls with structured audit entries."""

    job_id: str
    max_calls: int = 24
    _calls_used: int = 0
    _seen: Dict[str, str] = field(default_factory=dict)  # key -> result summary
    _tools: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._tools = {
            "web_search": WebSearchTool(),
            "python_sandbox": CodeExecutionTool(timeout=5),
            "sql_lookup": SQLLookupTool(),
            "self_reflection": SelfReflectionTool(),
        }

    def attach(self):
        return _tool_mediator_ctx.set(self)

    @staticmethod
    def detach(token) -> None:
        _tool_mediator_ctx.reset(token)

    @staticmethod
    def current() -> Optional["ToolMediator"]:
        return _tool_mediator_ctx.get()

    def _audit(
        self,
        context: SharedContext,
        tool_name: str,
        ok: bool,
        latency_ms: float,
        detail: Dict[str, Any],
        retry_count: int = 0,
    ) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "trace_id": context.job_id,
            "job_id": context.job_id,
            "agent_id": "tool_mediator",
            "event_type": "tool_call",
            "tool_name": tool_name,
            "success": ok,
            "latency_ms": latency_ms,
            "retry_count": retry_count,
            "detail": detail,
        }
        context.tool_audit.append(entry)
        context.execution_trace.append(
            {
                "timestamp": entry["timestamp"],
                "agent": "tool_mediator",
                "action": "tool_call",
                "details": entry,
                "job_id": context.job_id,
            }
        )

    async def invoke(
        self,
        context: SharedContext,
        tool_name: str,
        arg_factory: Callable[[], Dict[str, Any]],
        *,
        dedupe: bool = True,
    ) -> Tuple[bool, Any]:
        """
        arg_factory is a zero-arg callable so we only build heavy args if the call proceeds.
        """
        if self._calls_used >= self.max_calls:
            self._audit(
                context,
                tool_name,
                False,
                0.0,
                {"error": "tool_budget_exhausted", "max_calls": self.max_calls},
            )
            context.policy_violations.append(
                {"type": "tool_budget_exhausted", "tool": tool_name, "timestamp": datetime.now().isoformat()}
            )
            return False, None

        if tool_name not in self._tools:
            self._audit(context, tool_name, False, 0.0, {"error": "unknown_tool"})
            return False, None

        args = arg_factory()
        raw = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, default=str)
        key = hashlib.sha256(raw.encode()).hexdigest()
        if dedupe and key in self._seen:
            self._audit(context, tool_name, True, 0.0, {"deduped": True, "cache_key": key[:16]})
            return True, {"deduped": True, "summary": self._seen[key]}

        tool = self._tools[tool_name]
        start = asyncio.get_event_loop().time()
        result = await tool.call_with_retry(**args)
        latency_ms = (asyncio.get_event_loop().time() - start) * 1000
        self._calls_used += 1

        summary = "ok" if result.success else (result.error or "failed")
        if dedupe and result.success:
            self._seen[key] = summary

        self._audit(
            context,
            tool_name,
            result.success,
            latency_ms,
            {
                "args_preview": str(args)[:500],
                "error": result.error,
                "retry_count": getattr(result, "retry_count", 0),
            },
            retry_count=getattr(result, "retry_count", 0),
        )
        return result.success, result.data if result.success else None
