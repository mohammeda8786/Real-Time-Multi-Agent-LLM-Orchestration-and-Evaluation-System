"""Normalize persisted or in-memory job payloads for trace responses."""

from __future__ import annotations

from typing import Any, Dict, List


def _agent_value(obj: Any) -> str:
    if obj is None:
        return "unknown"
    if isinstance(obj, dict):
        v = obj.get("next_agent") or obj.get("agent")
        if isinstance(v, dict) and "value" in v:
            return str(v["value"])
        return str(v) if v is not None else "unknown"
    if hasattr(obj, "value"):
        return str(obj.value)
    return str(obj)


def _reasoning(obj: Any) -> str:
    if isinstance(obj, dict):
        return str(obj.get("reasoning") or "")
    return str(getattr(obj, "reasoning", "") or "")


def build_trace_decisions(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []
    for rd in result.get("routing_decisions") or []:
        decisions.append(
            {
                "agent": _agent_value(rd),
                "reasoning": _reasoning(rd),
                "source": "routing",
            }
        )
    for ev in result.get("execution_trace") or []:
        details = ev.get("details") if isinstance(ev, dict) else {}
        if isinstance(ev, dict) and ev.get("action") == "tool_call" and isinstance(details, dict):
            decisions.append(
                {
                    "agent": "tool_mediator",
                    "reasoning": f"tool={details.get('tool_name')} success={details.get('success')}",
                    "source": "tool_audit",
                    "latency_ms": details.get("latency_ms"),
                }
            )
        elif isinstance(ev, dict) and ev.get("action") == "agent_run":
            d = ev.get("details") or {}
            decisions.append(
                {
                    "agent": d.get("agent", "unknown"),
                    "reasoning": f"iteration={d.get('iteration')} status={d.get('status')}",
                    "source": "execution_trace",
                    "latency_ms": d.get("latency_ms"),
                }
            )
    return decisions


def normalize_chunk(c: Any) -> Dict[str, Any]:
    if isinstance(c, dict):
        return {
            "source": c.get("source"),
            "relevance": c.get("relevance_score"),
            "content": (c.get("content") or "")[:200],
            "chunk_id": c.get("chunk_id"),
        }
    return {
        "source": getattr(c, "source", None),
        "relevance": getattr(c, "relevance_score", None),
        "content": (getattr(c, "content", "") or "")[:200],
        "chunk_id": getattr(c, "chunk_id", None),
    }


def normalize_claim(cl: Any) -> Dict[str, Any]:
    if isinstance(cl, dict):
        text = cl.get("text") or ""
        return {
            "text": text[:150],
            "confidence": cl.get("confidence_score"),
            "citations": cl.get("chunk_citations"),
        }
    return {
        "text": (getattr(cl, "text", "") or "")[:150],
        "confidence": getattr(cl, "confidence_score", None),
        "citations": getattr(cl, "chunk_citations", None),
    }


def normalize_critique(cr: Any) -> Dict[str, Any]:
    if isinstance(cr, dict):
        return {
            "claim_id": cr.get("claim_id"),
            "disagreement": cr.get("disagreement_reason"),
            "suggestion": cr.get("suggested_correction"),
        }
    return {
        "claim_id": getattr(cr, "claim_id", None),
        "disagreement": getattr(cr, "disagreement_reason", None),
        "suggestion": getattr(cr, "suggested_correction", None),
    }
