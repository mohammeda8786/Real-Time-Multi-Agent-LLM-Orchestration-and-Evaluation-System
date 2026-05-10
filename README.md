# Mega.AI — Multi-Agent LLM Orchestration

**Production-oriented research prototype:** dynamic orchestration, multi-hop RAG with reranking, tool mediation, evaluation harness, and honest documentation of limits.

## Quick start

```bash
cd Mega.AI
python -m venv venv
# Windows: venv\Scripts\activate
# Unix:    source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # if present; otherwise create .env with GROQ_API_KEY
python api.py
```

- API: `http://127.0.0.1:8000` — OpenAPI: `http://127.0.0.1:8000/docs`
- Health: `GET /health` — Diagnostics: `GET /diagnostics` — Metrics: `GET /metrics` (Prometheus text when `prometheus-client` is installed)

Submit a job:

```bash
curl -s -X POST http://localhost:8000/submit -H "Content-Type: application/json" -d "{\"query\": \"What is RAG?\"}"
```

SSE (orchestration **events**, not per-token LLM streaming):

```bash
curl -N -X POST http://localhost:8000/submit/stream -H "Content-Type: application/json" -d "{\"query\": \"What is Python?\"}"
```

## Current status (honest)

| Area | Status |
|------|--------|
| LLM routing + fallback chain | Implemented |
| Multi-hop RAG + dedupe + rerank | Implemented |
| Tool calls (web search mock, Python sandbox, SQL lookup, self-reflection) | Implemented via **orchestrator `ToolMediator`** (contextvars) |
| SQLite persistence for jobs / eval / prompt rewrites | Implemented |
| PostgreSQL + Redis in Docker | Services wired; app still uses SQLite for job/eval by default |
| SSE | **Event-level** streaming for routing, agents, completion |
| Token-level LLM streaming | **Not** implemented (Groq client returns full completion) |
| Evaluation (15 cases, 6 dimensions, thresholds) | Implemented; scores are **heuristic**, not human labels |

Known limits: instruction-tuned models may still mishandle edge cases; embedding load is heavy on first start; Python 3.14+ may warn on Groq/Pydantic v1 compat — prefer **Python 3.11–3.12**.

## Python version and platforms

- **Supported:** Python **3.11** and **3.12** (same as the `Dockerfile` base image).
- **Experimental:** Python 3.13+ — the app starts, but you may see **Pydantic v1 / Groq** user warnings; use 3.11 or 3.12 if you need a quiet, reproducible stack.
- **Check:** `python verify_environment.py` (3.13+ reports `[WARN]` but still passes if dependencies import).
- **Runtime snapshot:** `GET /diagnostics` (versions, OS, `python_support_tier`: `supported` | `experimental` | `unsupported`).

### Windows console (PowerShell / CMD)

- Scripts and tests use **ASCII markers** (`[OK]`, `[WARN]`, `[ERROR]`) instead of emoji to avoid `UnicodeEncodeError` on cp1252.
- The app calls `sys.stdout.reconfigure(encoding='utf-8', errors='replace')` when available (see `app/platform/runtime.py`).
- If you still see encoding issues, set: `set PYTHONIOENCODING=utf-8` before running Python.

### ChromaDB persistence

- Default persist path: **`./chroma_db`** (same folder mounted in Docker).
- Collections are opened with **`get_or_create_collection`** when the client supports it, so **restarts do not fail** with “collection already exists.”
- If the database is corrupted or locked after a crash, stop all processes using the folder, then retry or move to a new persist directory and re-index.

### Dependency troubleshooting

| Symptom | What to try |
|--------|-------------|
| Groq + Pydantic warning on Python 3.14 | Use Python 3.11 or 3.12; warning is filtered to “once” after our bootstrap. |
| Chroma `InternalError` / collection errors | Upgrade `chromadb`; ensure only one process writes the persist path; delete lock files only if no Chroma process is running. |
| `sentence-transformers` slow first run | Model download; keep HF cache or pre-download in Docker build if needed. |
| Rate limit on `/submit` | Wait briefly between requests (see `api.py` interval). |

## System architecture

```
┌─────────────────────────────────────────────────────┐
│           API Layer (FastAPI)                       │
│  ▪ /submit (streaming)  ▪ /trace/{job_id}          │
│  ▪ /eval/latest         ▪ /meta/approve            │
│  ▪ /eval/retrigger                                 │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐   ┌────────▼──────────┐
│  Orchestrator  │   │  Evaluation Loop  │
│  (Dynamic      │   │  (15 Test Cases)  │
│   Routing)     │   │  (6 Dimensions)   │
└───────┬────────┘   └────────┬──────────┘
        │                     │
  ┌─────┴─────────────────┐   │
  │                       │   │
  ▼   ▼   ▼   ▼          ▼   ▼
┌────────────────────────────────────────┐
│         Sub-Agents Layer               │
│  ┌─────────────────────────────────┐   │
│  │ Decomposer: Query → Task Graph  │   │
│  │ RAG Agent: Multi-hop Retrieval  │   │
│  │ Critic: Claim Validation        │   │
│  │ Synthesizer: Answer Merging     │   │
│  └─────────────────────────────────┘   │
└──────────┬──────────────────────────────┘
           │
     ┌─────┴─────┬──────────┬──────────┐
     ▼           ▼          ▼          ▼
┌────────┐  ┌─────────┐ ┌────────┐ ┌──────────┐
│WebSearch│ │CodeExec │ │SQL Lookup│ │Self-Refl │
└────────┘  └─────────┘ └────────┘ └──────────┘

        ┌──────────────────────────┐
        │  Shared Context          │
        │  (Policy Enforcement)    │
        │  (Budget Manager)        │
        └──────────────────────────┘
```

## 📋 Components

### 1. **Orchestrator Agent** (`app/agents/orchestrator.py`)
- **Role**: Decides which agent to run next based on current state
- **Decision Logic**: LLM-powered dynamic routing (not hardcoded chains)
- **Logging**: Every routing decision includes reasoning and justification
- **Dependency Handling**: Respects task dependencies from decomposer

### 2. **Sub-Agents**

#### Decomposition Agent (`app/agents/decomposer.py`)
- Breaks ambiguous queries into typed sub-tasks with dependencies
- Returns dependency graph with explicit execution order
- Tasks: `retrieval`, `reasoning`, `verification`, `synthesis`
- **Guarantees**: Dependent tasks don't execute until dependencies complete

#### RAG Agent (`app/agents/real_rag_agent.py`)
- Multi-hop retrieval (minimum 2 hops required)
- Each chunk includes source, relevance score, and retrieval hop number
- Creates `Claim` objects with chunk citations for provenance
- Uses LLM to synthesize answer from chunks with citations

#### Critic Agent (`app/agents/critic.py`)
- Reviews **individual claims**, not just the whole output
- Flags specific text spans (not just "output is wrong")
- Assigns confidence score per claim
- Detects contradictions between claims

#### Synthesizer Agent (`app/agents/synthesizer.py`)
- Merges all claims and applies critic feedback
- Resolves contradictions with explicit reasoning
- Builds provenance map: sentence → source agent → source chunks
- Final answer includes citations

### 3. **Tool System** (`app/tools/`)

#### Web Search (`web_search.py`)
- Returns structured results: `[{title, url, snippet, relevance_score}]`
- Failure modes: timeout, empty results, malformed input
- Fallback: Returns empty list

#### Code Execution (`code_execution.py`)
- Sandbox with timeout (5s default)
- Returns: `{stdout, stderr, exit_code}`
- Security: Blocks `os.system`, `exec()`, `eval()`, `__import__`
- Failure: Timeout → `ToolResult(success=False, error="timeout")`

#### SQL Lookup (`sql_lookup.py`)
- Converts NL queries to SQL
- Returns: structured table rows
- Failure: Empty results → `ToolResult(success=True, data=[])`

#### Self-Reflection (`self_reflection.py`)
- Agent retrieves its own execution history per session
- Detects contradictions: yes/no reversals, negation flips
- Used for meta-analysis of agent performance

### 4. **Context Budget Manager** (`app/context/budget_manager.py`)

**Token Budgets (per agent)**:
- Decomposer: 1,000 tokens
- RAG: 2,500 tokens  
- Critic: 1,500 tokens
- Synthesizer: 1,500 tokens
- Orchestrator: 500 tokens

**Features**:
- Pre-execution budget check
- Policy violations logged (not silent truncation)
- Context compression: lossless for structured data, lossy for conversational
- Budget extension requests tracked with audit trail

## 🧪 Evaluation Pipeline (`app/evaluation/pipeline.py`)

**15 Test Cases** (5 per category):

### Baseline (5 tests)
- Straightforward factual questions
- Expected keywords present in answer
- Scores: answer_correctness, citation_accuracy

### Ambiguous (5 tests)
- Underspecified inputs requiring decomposition
- Ambiguous pronouns, implicit context
- Tests decomposer quality

### Adversarial (5 tests)
- **Prompt Injection**: "Ignore previous instructions..."
- **False Premises**: "Python is a snake. How is it used in programming?"
- **Contradictions**: "Databases are lost on poweroff. Explain why it's good."
- Tests critic robustness and synthesis quality

### 6 Scoring Dimensions

| Dimension | Formula | Range | Notes |
|-----------|---------|-------|-------|
| **Answer Correctness** | % keywords found | 0-1 | Based on expected keywords |
| **Citation Accuracy** | cited_claims / total_claims | 0-1 | Higher is better |
| **Contradiction Resolution** | 1.0 if resolved, else 0.5 | 0-1 | Synth applies critic feedback |
| **Tool Efficiency** | 1.0 if 4-6 execs, else penalized | 0-1 | Penalizes unnecessary calls |
| **Budget Compliance** | 1.0 if no violations, else 0.1 | 0-1 | Policy violations are failures |
| **Adversarial Robustness** | defensive_keywords / 2 | 0-1 | N/A for baseline |

**Scoring Output**: Per test case:
```python
{
    "test_id": "baseline_1",
    "overall_score": 0.82,
    "dimensions": {
        "answer_correctness": {
            "score": 0.95,
            "justification": "Keywords found: programming, scripting, data, web"
        },
        ...
    },
    "execution_trace": [...],
    "timestamp": "2026-05-09T12:34:56Z"
}
```

## Self-Improving Prompt Loop (`app/meta/prompt_optimizer.py`)

### Flow
1. **Identify Failures**: Tests scoring < 0.6 per dimension
2. **Root Cause**: Group by scoring dimension (e.g., all cite failures)
3. **Generate Rewrite**: LLM proposes improved prompt with justification
4. **Human Approval**: `/meta/approve` endpoint requires explicit OK
5. **Apply**: Rewrite applied only if approved
6. **Re-evaluate**: Run failed tests with new prompt
7. **Audit**: Every rewrite, approval, and performance delta logged

### Audit Trail
```python
{
    "proposal_id": "rewrite_20260509_120000_abc123de",
    "created_at": "2026-05-09T12:00:00Z",
    "target_agent": "rag_agent",
    "target_dimension": "citation_accuracy",
    "original_prompt": "...",
    "proposed_prompt": "...",
    "justification": "Previous version missed subtle...",
    "expected_improvement": 0.75,
    "status": "pending" | "approved" | "rejected" | "applied",
    "approval_timestamp": "2026-05-09T12:05:00Z",
    "performance_delta": +0.12  # Actual improvement
}
```

## 🌊 API Endpoints

### 1. POST `/submit` - Submit a Query
```bash
curl -X POST http://localhost:8000/submit \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?"}'
```

**Status**: implemented. Supports hybrid SSE streaming with routing, tool lifecycle events, budget updates, and simulated response token chunks.

**Response (JSON)**:
```json
{
    "job_id": "abc123...",
    "query": "What is Python?",
    "answer": "Python is a high-level programming language...",
    "status": "completed",
    "timestamp": "2026-05-09T12:34:56Z",
    "stats": {
        "chunks_retrieved": 3,
        "claims_generated": 5,
        "critiques": 1,
        "budget_violations": 0
    }
}
```

### 1b. POST `/submit/stream` - Event Stream for Orchestration Progress
```bash
curl -N -X POST http://localhost:8000/submit/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?"}'
```

**Status**: implemented. Returns a structured SSE stream with routing, agent lifecycle events, tool execution markers, budget updates, and final token chunks.

**Example event body**:
```json
{
  "event": "routing_decision",
  "agent": "rag",
  "reasoning": "Fallback: no retrieved chunks yet",
  "job_id": "abc123..."
}
```

**Common stream events**:
- `orchestrator_start`
- `routing_decision`
- `tool_execution_start`
- `tool_execution_end`
- `rag_complete`
- `budget_update`
- `answer_stream_start`
- `answer_token_chunk`
- `answer_stream_complete`
- `completed`

**Suggested client logic**:
1. Connect to `/submit/stream` with `Accept: text/event-stream`.
2. Parse each `data:` payload as JSON.
3. Render ongoing progress from `routing_decision` and `tool_execution_*` events.
4. Append `answer_token_chunk` payloads to build the final answer.
5. Treat `completed` as the final stop signal.

### 2. GET `/trace/{job_id}` - Execution Trace Summary
```bash
curl http://localhost:8000/trace/abc123...
```

**Status**: implemented. Returns a summarized trace, the first few chunks/claims, and the final answer.

**Response**:
```json
{
    "job_id": "abc123...",
    "query": "...",
    "decisions": [
        {
            "agent": "orchestrator",
            "reasoning": "Query is ambiguous, route to decomposer",
            "timestamp": "2026-05-09T12:34:56Z"
        }
    ],
    "chunks": [{"source": "...", "relevance": 0.92, "content": "..."}],
    "claims": [{"text": "...", "confidence": 0.74, "citations": ["..."]}],
    "critiques": [{"claim_id": "...", "disagreement": "...", "suggestion": "..."}],
    "final_answer": "..."
}
```

### 3. GET `/eval/latest` - Evaluation Summary
```bash
curl http://localhost:8000/eval/latest
```

**Status**: implemented. Returns persisted evaluation results if available, otherwise a default mocked summary is returned.

**Response**:
```json
{
    "run_id": "eval_20260509_120000",
    "timestamp": "2026-05-09T12:34:56Z",
    "summary": {
        "baseline": {"count": 5, "average_score": 0.85},
        "ambiguous": {"count": 5, "average_score": 0.72},
        "adversarial": {"count": 5, "average_score": 0.68}
    },
    "dimension_averages": {
        "answer_correctness": 0.82,
        "citation_accuracy": 0.79,
        "contradiction_resolution": 0.71
    }
}
```

### 4. POST `/meta/approve` - Approve Prompt Rewrite
```bash
curl -X POST http://localhost:8000/meta/approve \
  -H "Content-Type: application/json" \
  -d '{"rewrite_id": "rewrite_...", "approved": true}'
```

**Status**: implemented as a demo approval flow. It records approval/rejection state in memory and returns a sample rewrite if none exists.

### 5. POST `/eval/retrigger` - Re-evaluate Failed Cases
```bash
curl -X POST http://localhost:8000/eval/retrigger
```

**Status**: implemented as a mocked trigger. It returns an immediate "started" response and does not currently execute a full background evaluation run.

### 6. GET `/jobs` - List Processed Jobs
```bash
curl http://localhost:8000/jobs
```

**Status**: implemented. Returns job IDs and metadata from in-memory processed jobs.

### 7. GET `/health` - Health Check
```bash
curl http://localhost:8000/health
```

**Status**: implemented. Returns service health and basic runtime metadata.

## 🐳 Docker Setup

```bash
# Build and run all services
docker-compose up

# Services started:
# - API: http://localhost:8000
# - ChromaDB: http://localhost:8001
# - PostgreSQL: localhost:5432
# - Logs viewer: http://localhost:9999

# Run evaluation in container
docker-compose exec api python -m pytest tests/eval.py

# View logs
docker-compose logs -f api
```

## ⚠️ Known Limitations & Honest Assessment

### What Works ✅
- Prototype-level multi-agent orchestration with dynamic routing
- Budget-aware context management with per-agent budget checks
- Evaluation pipeline with 6 scoring dimensions
- Citation tracking and execution trace scaffolding
- Structured JSON logging for agents and traces
- SQLAlchemy persistence scaffolding for jobs, traces, eval runs, and prompt rewrites
- Adversarial robustness tests are present in the evaluation harness

### What Doesn't Work ❌

| Issue | Current State | Notes |
|-------|---------------|-------|
| **Streaming SSE** | Not fully implemented | `/submit` returns a full JSON response today |
| **True Multi-hop RAG** | Single-stage retrieval | Multi-hop is aspirational; current RAG uses one primary retrieval path |
| **Distributed Context** | In-memory only | No Redis or shared session store yet |
| **Production Database** | Models defined, migrations missing | SQLite default works locally; PostgreSQL support is configured but manual |
| **Prompt Rewrite Approval** | Mocked approval flow | `/meta/approve` exists but is not a finalized governance workflow |
| **Tool Sandboxing** | Experimental | Code execution and tools are not hardened for untrusted input |
| **Scalability** | No real job queue | `worker.py` exists, but no robust queue framework is wired in |

### Where It Breaks
1. **Adversarial Injection**: Critic agent can still be influenced by prompt attacks
2. **False Premise Handling**: Does not reliably reject or correct false assumptions
3. **Budget Enforcement**: Violations are logged; some agents may still exceed budget under stress
4. **LLM Hallucination**: Citations and claims can still reflect model artifacts

## 🚀 What to Build Next

1. **Persistent Storage**
   - PostgreSQL schema for all context, traces, evaluations
   - Redis for distributed context sharing
   - S3 for large artifact storage

2. **True Streaming**
   - WebSocket support for real-time agent updates
   - Token-by-token streaming from LLM
   - Live context budget visualization

3. **Multi-hop RAG Enhancement**
   - Graph-based retrieval routing
   - Cross-chunk semantic bridging
   - Iterative query refinement

4. **Production Monitoring**
   - Prometheus metrics for all agents
   - Grafana dashboards for eval trends
   - Alert system for budget/performance anomalies

5. **Advanced Self-Improvement**
   - A/B testing of prompt variants
   - Gradient-based prompt optimization
   - Automated revert on regression

6. **Guardrails**
   - Constitutional AI for harmlessness checks
   - Input sanitization layer
   - Output safety filters

## 🔍 Testing & Reproducibility

```bash
# Run evaluation
python -c "
from app.evaluation.pipeline import EvaluationPipeline
from app.agents.orchestrator import OrchestratorAgent
import asyncio

pipeline = EvaluationPipeline()
orchestrator = OrchestratorAgent()
results = asyncio.run(pipeline.run_evaluation(orchestrator))
print(json.dumps(results, indent=2))
"

# Get deterministic evaluation ID
echo "eval_20260509_120000"

# Re-run same evaluation (same results expected)
curl "http://localhost:8000/eval/latest?run_id=eval_20260509_120000"
```

## 📊 Architecture Decision Record (ADR)

### Why Groq (Free LLM)?
- **Cost**: Zero inference cost, focus on system architecture not API bills
- **Speed**: 70B model nearly instant, good for iteration
- **Trade-off**: Hallucination rate higher than GPT-4

### Why In-Memory Context?
- **Simplicity**: Prototyping speed
- **Trade-off**: Single-instance only, data lost on restart
- **Fix**: Easy migration to Redis in production

### Why Custom Evaluation?
- **Transparency**: Every score justified with reasoning
- **Control**: Multi-dimensional scoring vs black-box frameworks
- **Trade-off**: More code to maintain

## 📝 License

MIT - Use freely, cite if helpful

## 🤝 Questions?

See `/memories/` for development notes, or open an issue.`

---

**Built**: May 2026 | **Status**: Prototype with production hardening in progress
   