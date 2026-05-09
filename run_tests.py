"""
Complete Test Suite - Verifies all system components
Run: python run_tests.py
"""

import asyncio
import json
import sys
from datetime import datetime

from app.platform.runtime import configure_runtime_warnings, configure_stdio_utf8

configure_stdio_utf8()
configure_runtime_warnings()

print("\n" + "="*60)
print("MEGA.AI - COMPLETE TEST SUITE")
print("="*60 + "\n")

# Test 1: Import all modules
print("[TEST 1/5] Module Imports...")
try:
    from app.agents.orchestrator import OrchestratorAgent
    from app.agents.decomposer import DecompositionAgent
    from app.agents.real_rag_agent import RealRAGAgent
    from app.agents.critic import CriticAgent
    from app.agents.synthesizer import SynthesizerAgent
    from app.models.schemas import SharedContext, AgentType
    from app.evaluation.pipeline import EvaluationPipeline
    from app.meta.prompt_optimizer import SelfImprovingPromptLoop
    from app.tools import WebSearchTool, CodeExecutionTool, SQLLookupTool, SelfReflectionTool
    from app.context.budget_manager import ContextBudgetManager
    from app.llm_client import LLMClient
    print("[OK] All imports successful\n")
except Exception as e:
    print(f"[ERROR] Import failed: {e}\n")
    sys.exit(1)

# Test 2: LLM Client
print("[TEST 2/5] LLM Client...")
try:
    llm = LLMClient()
    text = "Hello, world!"
    tokens = llm.count_tokens(text)
    print(f"[OK] LLM client ready (token count: {tokens})\n")
except Exception as e:
    print(f"[ERROR] LLM client failed: {e}\n")

# Test 3: Tool System
print("[TEST 3/5] Tool System (with retry)...")
try:
    web_search = WebSearchTool()
    code_exec = CodeExecutionTool()
    sql_lookup = SQLLookupTool()
    self_reflection = SelfReflectionTool()
    
    # Test web search
    result = asyncio.run(web_search.call_with_retry(query="Python programming"))
    assert result.success, "Web search failed"
    assert len(result.data["results"]) > 0, "No search results"
    print(f"[OK] Web search: {len(result.data['results'])} results")
    
    # Test code execution (safe code)
    result = asyncio.run(code_exec.call_with_retry(code="print('Hello')"))
    assert result.success, "Code execution failed"
    print(f"[OK] Code execution: {result.data['exit_code']} (success)")
    
    # Test SQL lookup
    result = asyncio.run(sql_lookup.call_with_retry(nl_query="languages"))
    assert result.success, "SQL lookup failed"
    print(f"[OK] SQL lookup: {len(result.data.get('results', []))} entities")
    
    # Test self-reflection
    self_reflection.record_execution("agent1", "session1", "output1")
    result = asyncio.run(self_reflection.call_with_retry(agent_id="agent1", session_id="session1"))
    assert result.success, "Self-reflection failed"
    print(f"[OK] Self-reflection: {result.data['execution_count']} recorded\n")
    
except Exception as e:
    print(f"[ERROR] Tool system failed: {e}\n")
    import traceback
    traceback.print_exc()

# Test 4: Budget Manager
print("[TEST 4/5] Budget Manager...")
try:
    budget_mgr = ContextBudgetManager()
    context = SharedContext(original_query="Test query")
    asyncio.run(budget_mgr.initialize_budgets(context))
    
    # Check budgets initialized
    assert len(context.budgets) > 0, "No budgets initialized"
    
    # Test deduction
    success = asyncio.run(budget_mgr.deduct_budget(context, AgentType.RAG, 100))
    assert success, "Budget deduction failed"
    
    remaining = asyncio.run(budget_mgr.get_remaining_budget(context, AgentType.RAG))
    assert remaining == 2400, f"Expected 2400 remaining, got {remaining}"
    
    report = budget_mgr.get_budget_report(context)
    print(f"[OK] Budget manager: {len(context.budgets)} agents tracked")
    print(f"   Total: {report['total_allocated']} allocated, {report['total_used']} used")
    print(f"   RAG utilization: {report['per_agent'][AgentType.RAG]['utilization_percent']:.1f}%\n")
    
except Exception as e:
    print(f"[ERROR] Budget manager failed: {e}\n")
    import traceback
    traceback.print_exc()

# Test 5: Evaluation Pipeline
print("[TEST 5/5] Evaluation Pipeline...")
try:
    eval_pipeline = EvaluationPipeline()
    
    # Check test cases created
    assert len(eval_pipeline.test_cases) == 15, f"Expected 15 tests, got {len(eval_pipeline.test_cases)}"
    
    baseline = [t for t in eval_pipeline.test_cases if t.category == "baseline"]
    ambiguous = [t for t in eval_pipeline.test_cases if t.category == "ambiguous"]
    adversarial = [t for t in eval_pipeline.test_cases if t.category == "adversarial"]
    
    print(f"✅ Test cases created:")
    print(f"   • {len(baseline)} baseline tests")
    print(f"   • {len(ambiguous)} ambiguous tests")
    print(f"   • {len(adversarial)} adversarial tests")
    print(f"\n   Sample tests:")
    for test in eval_pipeline.test_cases[:3]:
        print(f"   - {test.id}: {test.query[:50]}...")
    print()
    
except Exception as e:
    print(f"[ERROR] Evaluation pipeline failed: {e}\n")
    import traceback
    traceback.print_exc()

print("="*60)
print("TEST SUMMARY")
print("="*60)
print("""
[OK] All components verified (see messages above for any [ERROR] lines).

Next Steps:
1. START API SERVER:
   python api.py

2. TEST ENDPOINTS (in another terminal):
   
   a) Health Check:
      curl http://localhost:8000/health
   
   b) Submit Query:
      curl -X POST http://localhost:8000/submit \\
        -H "Content-Type: application/json" \\
        -d '{"query": "What is Python used for?"}'
   
   c) Get Trace:
      curl http://localhost:8000/trace/{job_id}
   
   d) Get Evaluation Summary:
      curl http://localhost:8000/eval/latest
   
   e) Approve Rewrite:
      curl -X POST http://localhost:8000/meta/approve \\
        -H "Content-Type: application/json" \\
        -d '{"rewrite_id": "test", "approved": true}'

3. RUN FULL EVALUATION:
   python run_evaluation.py

4. VIEW API DOCS:
   http://localhost:8000/docs

5. DEPLOY WITH DOCKER:
   docker-compose up

""")
print("="*60 + "\n")
