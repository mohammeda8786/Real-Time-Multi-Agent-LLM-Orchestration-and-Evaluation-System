"""
Debug script to identify the orchestrator failure
"""

import asyncio
import traceback
from app.agents.orchestrator import OrchestratorAgent
from app.models.schemas import SharedContext

async def main():
    try:
        print("Initializing orchestrator...")
        orchestrator = OrchestratorAgent()
        
        print("Creating shared context...")
        context = SharedContext(original_query='What is the difference between reinforcement learning and supervised learning?')
        
        print("Running orchestrator process...")
        result = await orchestrator.process(context)
        
        print("\n[OK] SUCCESS")
        print(f"Status: {result.status}")
        print(f"Answer: {result.synthesized_answer[:100] if result.synthesized_answer else 'None'}")
        print(f"Claims: {len(result.claims)}")
        print(f"Chunks: {len(result.retrieved_chunks)}")
        print(f"Violations: {len(result.policy_violations)}")
        print(f"Routing decisions: {len(result.routing_decisions)}")
        
        if result.policy_violations:
            print("\nPolicy violations:")
            for v in result.policy_violations:
                print(f"  - {v}")
        
        return result
        
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    asyncio.run(main())
