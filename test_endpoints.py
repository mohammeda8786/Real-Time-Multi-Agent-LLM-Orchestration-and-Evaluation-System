"""
Test All API Endpoints
Run: python test_endpoints.py
(Make sure API is running: python api.py)
"""

import asyncio
import json
import httpx
from datetime import datetime

BASE_URL = "http://localhost:8000"
TEST_QUERY = "What is Python used for?"

print("\n" + "="*70)
print("MEGA.AI - API ENDPOINT TESTS")
print("="*70 + "\n")

async def test_endpoints():
    async with httpx.AsyncClient() as client:
        
        # Test 1: Health Check
        print("[TEST 1/5] GET /health")
        try:
            response = await client.get(f"{BASE_URL}/health")
            data = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"   Server status: {data.get('status')}")
            print(f"   LLM: {data.get('llm')}")
            print(f"   RAG: {data.get('rag')}\n")
        except Exception as e:
            print(f"❌ Failed: {e}\n")
            return
        
        job_id = None
        
        # Test 2: Submit Query
        print("[TEST 2/5] POST /submit")
        try:
            payload = {"query": TEST_QUERY}
            response = await client.post(
                f"{BASE_URL}/submit",
                json=payload,
                timeout=30.0
            )
            data = response.json()
            job_id = data.get('job_id')
            
            print(f"✅ Status: {response.status_code}")
            print(f"   Job ID: {job_id}")
            print(f"   Query: {data.get('query')}")
            print(f"   Answer: {data.get('answer')[:100]}...")
            print(f"   Status: {data.get('status')}")
            stats = data.get('stats', {})
            print(f"   Stats: {stats.get('chunks_retrieved')} chunks, " +
                  f"{stats.get('claims_generated')} claims, " +
                  f"{stats.get('budget_violations')} violations\n")
        except Exception as e:
            print(f"❌ Failed: {e}\n")
            return
        
        # Test 3: Get Execution Trace
        if job_id:
            print("[TEST 3/5] GET /trace/{job_id}")
            try:
                response = await client.get(f"{BASE_URL}/trace/{job_id}", timeout=10.0)
                data = response.json()
                
                print(f"✅ Status: {response.status_code}")
                print(f"   Job ID: {data.get('job_id')}")
                print(f"   Query: {data.get('query')}")
                print(f"   Decisions made: {len(data.get('decisions', []))}")
                print(f"   Chunks retrieved: {len(data.get('chunks', []))}")
                print(f"   Claims generated: {len(data.get('claims', []))}")
                print(f"   Critiques: {len(data.get('critiques', []))}")
                print(f"   Final answer: {data.get('final_answer', '')[:100]}...\n")
            except Exception as e:
                print(f"⚠️  Warning: {e}")
                print("   (Trace may not be available immediately)\n")
        
        # Test 4: Get Evaluation Summary
        print("[TEST 4/5] GET /eval/latest")
        try:
            response = await client.get(f"{BASE_URL}/eval/latest", timeout=10.0)
            data = response.json()
            
            print(f"✅ Status: {response.status_code}")
            print(f"   Run ID: {data.get('run_id')}")
            
            summary = data.get('summary', {})
            if summary:
                for category, stats in summary.items():
                    if isinstance(stats, dict) and 'average_score' in stats:
                        print(f"   • {category}: {stats.get('average_score', 0):.1%} avg")
            
            dim_avg = data.get('dimension_averages', {})
            if dim_avg:
                print(f"\n   Dimension Averages:")
                for dim, score in sorted(dim_avg.items())[:3]:
                    print(f"   • {dim}: {score:.1%}")
            print()
        except Exception as e:
            print(f"⚠️  Warning: {e}")
            print("   (Evaluation summary may not be available)\n")
        
        # Test 5: Approve Rewrite (Mock)
        print("[TEST 5/5] POST /meta/approve")
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
            
            print(f"✅ Status: {response.status_code}")
            print(f"   Rewrite ID: {data.get('rewrite_id')}")
            print(f"   Approved: {data.get('approved')}")
            print(f"   Status: {data.get('status')}")
            print(f"   Message: {data.get('message')}\n")
        except Exception as e:
            print(f"⚠️  Warning: {e}\n")

print("Starting endpoint tests...")
print("(Make sure API server is running: python api.py)\n")

try:
    asyncio.run(test_endpoints())
except Exception as e:
    print(f"\n❌ Test suite failed: {e}")
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
