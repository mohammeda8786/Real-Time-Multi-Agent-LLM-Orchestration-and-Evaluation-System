"""
Test All API Endpoints
Run: python test_endpoints.py
(Make sure API is running: python api.py)
"""

import asyncio
import json
from datetime import datetime

import httpx

from app.platform.runtime import configure_runtime_warnings, configure_stdio_utf8

configure_stdio_utf8()
configure_runtime_warnings()

BASE_URL = "http://localhost:8000"
TEST_QUERY = "What is Python used for?"

print("\n" + "="*70)
print("MEGA.AI - API ENDPOINT TESTS")
print("="*70 + "\n")

async def test_endpoints():
    async with httpx.AsyncClient() as client:
        
        # Test 1: Diagnostics
        print("[TEST 1/6] GET /diagnostics")
        try:
            response = await client.get(f"{BASE_URL}/diagnostics", timeout=10.0)
            data = response.json()
            print(f"[OK] Status: {response.status_code}")
            print(f"   python: {data.get('python_version')} ({data.get('python_support_tier')})")
            print(f"   groq: {data.get('groq_version')} chromadb: {data.get('chromadb_version')} pydantic: {data.get('pydantic_version')}\n")
        except Exception as e:
            print(f"[ERROR] Failed: {e}\n")
            return

        # Test 2: Health Check
        print("[TEST 2/6] GET /health")
        try:
            response = await client.get(f"{BASE_URL}/health")
            data = response.json()
            print(f"[OK] Status: {response.status_code}")
            print(f"   Server status: {data.get('status')}")
            print(f"   python_support_tier: {data.get('python_support_tier')}")
            print(f"   jobs_in_memory: {data.get('jobs_in_memory')}\n")
        except Exception as e:
            print(f"[ERROR] Failed: {e}\n")
            return
        
        job_id = None
        
        # Test 3: Submit Query
        print("[TEST 3/6] POST /submit")
        try:
            payload = {"query": TEST_QUERY}
            response = await client.post(
                f"{BASE_URL}/submit",
                json=payload,
                timeout=30.0
            )
            data = response.json()
            job_id = data.get('job_id')
            
            print(f"[OK] Status: {response.status_code}")
            print(f"   Job ID: {job_id}")
            print(f"   Query: {data.get('query')}")
            print(f"   Answer: {data.get('answer')[:100]}...")
            print(f"   Status: {data.get('status')}")
            stats = data.get('stats', {})
            print(f"   Stats: {stats.get('chunks_retrieved')} chunks, " +
                  f"{stats.get('claims_generated')} claims, " +
                  f"{stats.get('budget_violations')} violations\n")
        except Exception as e:
            print(f"[ERROR] Failed: {e}\n")
            return
        
        # Test 4: Get Execution Trace
        if job_id:
            print("[TEST 4/6] GET /trace/{job_id}")
            try:
                response = await client.get(f"{BASE_URL}/trace/{job_id}", timeout=10.0)
                data = response.json()
                
                print(f"[OK] Status: {response.status_code}")
                print(f"   Job ID: {data.get('job_id')}")
                print(f"   Query: {data.get('query')}")
                print(f"   Decisions made: {len(data.get('decisions', []))}")
                print(f"   Chunks retrieved: {len(data.get('chunks', []))}")
                print(f"   Claims generated: {len(data.get('claims', []))}")
                print(f"   Critiques: {len(data.get('critiques', []))}")
                print(f"   Final answer: {data.get('final_answer', '')[:100]}...\n")
            except Exception as e:
                print(f"[WARN] {e}")
                print("   (Trace may not be available immediately)\n")
        
        # Test 5: Get Evaluation Summary
        print("[TEST 5/6] GET /eval/latest")
        try:
            response = await client.get(f"{BASE_URL}/eval/latest", timeout=10.0)
            data = response.json()
            
            print(f"[OK] Status: {response.status_code}")
            print(f"   Run ID: {data.get('run_id')}")
            
            summary = data.get('summary', {})
            if summary and isinstance(summary, dict):
                for category, stats in summary.items():
                    if isinstance(stats, dict) and 'average_score' in stats:
                        print(f"   - {category}: {stats.get('average_score', 0):.1%} avg")
            
            dim_avg = data.get('dimension_averages', data.get('overall_scores', {}))
            if dim_avg:
                print(f"\n   Dimension averages (sample):")
                for dim, score in sorted(dim_avg.items())[:3]:
                    print(f"   - {dim}: {score:.1%}")
            print()
        except Exception as e:
            print(f"[WARN] {e}")
            print("   (Evaluation summary may not be available)\n")
        
        # Test 6: Approve Rewrite (Mock)
        print("[TEST 6/6] POST /meta/approve")
        try:
            payload = {
                "rewrite_id": "rewrite_test_001",
                "approved": True
            }
            response = await client.post(
                f"{BASE_URL}/meta/approve",
                json=payload,
                timeout=10.0
            )
            data = response.json()
            
            print(f"[OK] Status: {response.status_code}")
            print(f"   Rewrite ID: {data.get('rewrite_id')}")
            print(f"   Approved: {data.get('approved')}")
            print(f"   Status: {data.get('status')}\n")
        except Exception as e:
            print(f"[WARN] {e}\n")

print("Starting endpoint tests...")
print("(Make sure API server is running: python api.py)\n")

try:
    asyncio.run(test_endpoints())
except Exception as e:
    print(f"\n[ERROR] Test suite failed: {e}")
    print("\nMake sure the API server is running:")
    print("  python api.py")

print("\n" + "="*70)
print("ENDPOINT TEST COMPLETE")
print("="*70)
print("""
API Endpoints Summary:

1. GET /health
   ✓ Server health check
   ✓ Returns: status, llm, rag

2. POST /submit
   ✓ Submit query to orchestrator
   ✓ Runs full agent pipeline
   ✓ Returns: job_id, answer, stats

3. GET /trace/{job_id}
   ✓ Get full execution trace
   ✓ Returns: decisions, chunks, claims, critiques

4. GET /eval/latest
   ✓ Get evaluation summary
   ✓ Returns: scores by category, dimension averages

5. POST /meta/approve
   ✓ Approve/reject prompt rewrites
   ✓ Human-in-the-loop improvement

Additional endpoints:
• GET /  - API information
• GET /jobs - List all processed jobs
• POST /eval/retrigger - Re-evaluate failed tests

For interactive testing:
  curl http://localhost:8000/docs
  (Open in browser for Swagger UI)
""")
print("="*70 + "\n")
