"""
Multi-agent LLM orchestration API: submit, traces, evaluation hooks, SSE, lightweight metrics.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional

from app.platform.runtime import (
    configure_runtime_warnings,
    configure_stdio_utf8,
    gather_runtime_diagnostics,
    log_startup_stage,
    stage_timer,
    warn_unsupported_python,
)

configure_stdio_utf8()
configure_runtime_warnings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warn_unsupported_python()

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.agents.orchestrator import OrchestratorAgent
from app.api_trace import (
    build_trace_decisions,
    normalize_chunk,
    normalize_claim,
    normalize_critique,
)
from app.evaluation.pipeline import EvaluationPipeline
from app.models.schemas import SharedContext
from app.persistence import PersistenceStore

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    PROM_JOBS = Counter("mega_ai_jobs_total", "Total jobs processed")
    PROM_JOB_LATENCY = Histogram("mega_ai_job_latency_seconds", "Job wall time")
    PROM_ENABLED = True
except ImportError:
    PROM_ENABLED = False
    PROM_JOBS = None  # type: ignore
    PROM_JOB_LATENCY = None  # type: ignore
    CONTENT_TYPE_LATEST = "text/plain"
    generate_latest = lambda: b"# prometheus_client not installed\n"

def _python_support_tier() -> str:
    if sys.version_info < (3, 11):
        return "unsupported"
    if sys.version_info >= (3, 13):
        return "experimental"
    return "supported"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    t0 = stage_timer()
    log_startup_stage("api_lifespan_start", detail="FastAPI lifespan begin")
    diag = gather_runtime_diagnostics()
    logger.info("startup_diagnostics", extra=diag)
    log_startup_stage(
        "api_listening",
        latency_ms=(stage_timer() - t0) * 1000,
        detail="Orchestrator loads lazily on first /submit",
    )
    yield
    log_startup_stage("api_shutdown", detail="process exiting")


app = FastAPI(
    title="Mega.AI Multi-Agent Orchestration",
    version="1.1.0",
    description="Research-oriented orchestration platform with RAG, evaluation, and observability hooks.",
    lifespan=_lifespan,
)

_orchestrator: Optional[OrchestratorAgent] = None


def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator
    if _orchestrator is None:
        t0 = stage_timer()
        _orchestrator = OrchestratorAgent()
        log_startup_stage(
            "orchestrator_initialized",
            latency_ms=(stage_timer() - t0) * 1000,
            detail="Includes RAG pipeline, Chroma, embedding model",
        )
    return _orchestrator


job_results: Dict[str, dict] = {}
eval_results: Dict[str, dict] = {}
pending_rewrites: List[dict] = []
store = PersistenceStore()

_last_submit: Dict[str, float] = defaultdict(float)
_MIN_SUBMIT_INTERVAL_S = 0.25


class QueryRequest(BaseModel):
    query: str
    job_id: Optional[str] = None


class QueryResponse(BaseModel):
    job_id: str
    query: str
    answer: str
    status: str
    timestamp: str
    stats: Optional[dict] = None


class TraceResponse(BaseModel):
    job_id: str
    query: str
    decisions: List[dict]
    chunks: List[dict]
    claims: List[dict]
    critiques: List[dict]
    final_answer: str
    tool_audit: List[dict] = Field(default_factory=list)
    multi_hop: List[dict] = Field(default_factory=list)


class ApproveRequest(BaseModel):
    rewrite_id: str
    approved: bool


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _rate_limit(request: Request) -> None:
    key = _client_key(request)
    now = time.monotonic()
    last = _last_submit[key]
    if now - last < _MIN_SUBMIT_INTERVAL_S:
        raise HTTPException(status_code=429, detail="Rate limit: wait before submitting again")
    _last_submit[key] = now


@app.post("/submit", response_model=QueryResponse)
async def submit_query(request: QueryRequest, req: Request):
    _rate_limit(req)
    job_id = request.job_id or str(uuid.uuid4())
    logger.info("submit job_id=%s", job_id)

    t0 = time.monotonic()
    context = SharedContext(original_query=request.query)
    context.job_id = job_id

    result = await get_orchestrator().process(context)
    elapsed = time.monotonic() - t0
    if PROM_ENABLED and PROM_JOBS and PROM_JOB_LATENCY:
        PROM_JOBS.inc()
        PROM_JOB_LATENCY.observe(elapsed)

    payload_result = result.model_dump(mode="json")
    job_payload = {
        "query": request.query,
        "result": payload_result,
        "timestamp": datetime.now().isoformat(),
    }
    job_results[job_id] = job_payload
    store.save_job_result(job_id, request.query, payload_result, result.status)

    return QueryResponse(
        job_id=job_id,
        query=request.query,
        answer=result.synthesized_answer or "No answer generated",
        status=result.status,
        timestamp=datetime.now().isoformat(),
        stats={
            "chunks_retrieved": len(result.retrieved_chunks),
            "claims_generated": len(result.claims),
            "critiques": len(result.critiques),
            "budget_violations": len(result.policy_violations),
            "tool_calls": len(result.tool_audit),
            "latency_s": round(elapsed, 3),
        },
    )


@app.post("/submit/stream")
async def submit_stream(request: QueryRequest, req: Request):
    """Server-Sent Events: routing, agent/tool events, completion (not token-level LLM streaming)."""
    _rate_limit(req)
    job_id = request.job_id or str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        async def streaming_callback(event_type: str, payload: dict):
            await queue.put({"event": event_type, **payload})

        async def run_job():
            try:
                ctx = SharedContext(original_query=request.query)
                ctx.job_id = job_id
                out = await get_orchestrator().process(ctx, streaming_callback)
                await queue.put(
                    {
                        "event": "completed",
                        "job_id": job_id,
                        "status": out.status,
                        "answer": out.synthesized_answer,
                        "chunks": len(out.retrieved_chunks),
                        "claims": len(out.claims),
                    }
                )
                payload_result = out.model_dump(mode="json")
                job_results[job_id] = {
                    "query": request.query,
                    "result": payload_result,
                    "timestamp": datetime.now().isoformat(),
                }
                store.save_job_result(job_id, request.query, payload_result, out.status)
            except Exception as e:
                await queue.put({"event": "error", "message": str(e), "job_id": job_id})
            finally:
                done.set()

        task = asyncio.create_task(run_job())
        heartbeat = time.monotonic()
        try:
            while not done.is_set() or not queue.empty():
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=12.0)
                    yield f"data: {json.dumps(item, default=str)}\n\n"
                except asyncio.TimeoutError:
                    now = time.monotonic()
                    if now - heartbeat > 12.0:
                        heartbeat = now
                        yield f"data: {json.dumps({'event': 'heartbeat', 'job_id': job_id})}\n\n"
                    if done.is_set():
                        break
            await task
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/trace/{job_id}", response_model=TraceResponse)
async def get_trace(job_id: str):
    if job_id not in job_results:
        saved = store.get_job(job_id)
        if not saved:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        job_results[job_id] = {
            "query": saved["query"],
            "result": saved["result"],
            "timestamp": saved["updated_at"],
        }
    data = job_results[job_id]
    result = data["result"]

    decisions = build_trace_decisions(result)
    chunks = [normalize_chunk(c) for c in (result.get("retrieved_chunks") or [])[:8]]
    claims = [normalize_claim(c) for c in (result.get("claims") or [])[:8]]
    critiques = [normalize_critique(c) for c in (result.get("critiques") or [])]

    return TraceResponse(
        job_id=job_id,
        query=data["query"],
        decisions=decisions,
        chunks=chunks,
        claims=claims,
        critiques=critiques,
        final_answer=result.get("synthesized_answer") or "",
        tool_audit=list(result.get("tool_audit") or [])[:50],
        multi_hop=list(result.get("multi_hop_reasoning_path") or [])[:10],
    )


@app.get("/eval/latest")
async def get_latest_eval():
    latest = store.get_latest_evaluation()
    if latest:
        return latest
    if eval_results:
        return list(eval_results.values())[-1]
    return {
        "status": "none",
        "message": "No evaluation has been persisted yet. Run `python run_evaluation.py` or POST /eval/retrigger.",
    }


@app.post("/eval/retrigger")
async def retrigger_evaluation(background_tasks: BackgroundTasks):
    async def _job():
        try:
            pipeline = EvaluationPipeline()
            summary = await pipeline.run_evaluation(get_orchestrator())
            eval_results[summary["eval_id"]] = summary
            logger.info("evaluation finished %s", summary["eval_id"])
        except Exception as exc:
            logger.exception("evaluation failed: %s", exc)
            eval_results[f"eval_err_{datetime.now().strftime('%H%M%S')}"] = {
                "status": "failed",
                "error": str(exc),
            }

    background_tasks.add_task(_job)
    return {
        "status": "started",
        "message": "Evaluation scheduled in background; poll GET /eval/latest",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/meta/approve")
async def approve_rewrite(request: ApproveRequest):
    row = store.get_prompt_rewrite(request.rewrite_id)
    if not row:
        raise HTTPException(status_code=404, detail="Unknown proposal id; run evaluation to generate proposals")
    row["status"] = "approved" if request.approved else "rejected"
    row["approval_timestamp" if request.approved else "rejected_at"] = datetime.now().isoformat()
    store.save_prompt_rewrite(row)
    return {
        "rewrite_id": request.rewrite_id,
        "approved": request.approved,
        "status": row["status"],
        "proposal": row,
    }


@app.get("/metrics")
async def metrics():
    if not PROM_ENABLED:
        return PlainTextResponse("# prometheus_client not installed\n", media_type="text/plain")
    data = generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    return {
        "service": "Mega.AI Multi-Agent Orchestration",
        "version": "1.1.0",
        "docs": "/docs",
        "health": "/health",
        "diagnostics": "/diagnostics",
        "metrics": "/metrics",
        "endpoints": [
            {"method": "POST", "path": "/submit", "description": "Run orchestration (sync response)"},
            {"method": "POST", "path": "/submit/stream", "description": "SSE event stream for a job"},
            {"method": "GET", "path": "/trace/{job_id}", "description": "Execution trace + tool audit"},
            {"method": "GET", "path": "/eval/latest", "description": "Last evaluation summary"},
            {"method": "POST", "path": "/eval/retrigger", "description": "Schedule evaluation run"},
            {"method": "POST", "path": "/meta/approve", "description": "Approve prompt rewrite proposal"},
            {"method": "GET", "path": "/diagnostics", "description": "Runtime and dependency versions"},
        ],
    }


@app.get("/diagnostics")
async def diagnostics():
    """Runtime environment snapshot (safe on all platforms; ASCII-only JSON)."""
    data = gather_runtime_diagnostics()
    data["python_support_tier"] = _python_support_tier()
    data["timestamp"] = datetime.now().isoformat()
    return data


@app.get("/health")
async def health():
    orchestrator_ready = False
    status = "degraded"
    try:
        orchestrator = get_orchestrator()
        orchestrator_ready = orchestrator.is_ready()
        status = "healthy" if orchestrator_ready else "degraded"
    except Exception as exc:
        status = "degraded"
        logger.warning("health_check_orchestrator_failed", extra={"error": str(exc)})

    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "orchestrator_ready": orchestrator_ready,
        "jobs_in_memory": len(job_results),
        "python_support_tier": _python_support_tier(),
    }


@app.get("/jobs")
async def list_jobs():
    return {
        "total_jobs": len(job_results),
        "job_ids": list(job_results.keys()),
        "jobs": [
            {"job_id": jid, "query": data["query"], "timestamp": data["timestamp"]}
            for jid, data in job_results.items()
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
