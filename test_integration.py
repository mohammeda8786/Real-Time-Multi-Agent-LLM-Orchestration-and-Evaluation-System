# test_integration.py
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agents.orchestrator import OrchestratorAgent
from app.models.schemas import SharedContext

async def main():
    print("="*60)
    print("INTEGRATION TEST: Orchestrator + REAL RAG")
    print("="*60)
    
    orchestrator = OrchestratorAgent()
    
    # Test queries
    test_queries = [
        "What is Python used for?",
        "Compare reinforcement learning and supervised learning"
    ]
    
    for query in test_queries:
        print(f"\n{'─'*50}")
        print(f"QUERY: {query}")
        print(f"{'─'*50}")
        
        context = SharedContext(original_query=query)
        
        async def callback(event_type, data):
            if event_type == "routing_decision":
                print(f"  → {data.get('agent')}: {data.get('reasoning', '')[:50]}...")
            elif event_type == "rag_complete":
                print(f"  RAG: {data.get('chunks_retrieved')} chunks, {data.get('claims_generated')} claims")
            elif event_type == "synthesizer_complete":
                print(f"  Final answer ready")
        
        result = await orchestrator.process(context, callback)
        
        print(f"\nFinal Answer Preview: {result.synthesized_answer[:300] if result.synthesized_answer else 'None'}...")
        print(f"Retrieved Chunks: {len(result.retrieved_chunks)}")
        print(f"Claims Generated: {len(result.claims)}")

if __name__ == "__main__":
    asyncio.run(main())