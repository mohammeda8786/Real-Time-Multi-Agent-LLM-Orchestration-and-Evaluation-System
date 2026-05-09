import asyncio
import sys
import os
from app.agents.orchestrator import OrchestratorAgent
from app.models.schemas import SharedContext 

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.agents.orchestrator import OrchestratorAgent
from app.models.schemas import SharedContext

async def main():
    # Initialize orchestrator
    orchestrator = OrchestratorAgent()
    
    # Test query that requires all agents
    query = "Compare the effectiveness of reinforcement learning versus supervised learning for robotics applications"
    
    # Create context
    context = SharedContext(original_query=query)
    result = await orchestrator.process(context)
    print(result.synthesized_answer)
    
    # Define streaming callback
    async def stream_callback(event_type: str, data: dict):
        # No asyncio.sleep needed here
        message = data.get('message', data)
        if 'token' in data:
            # For token streaming, print without newline
            print(data['token'], end='', flush=True)
        else:
            print(f"\n[EVENT] {event_type}: {message}")
    
    try:
        # Run orchestration
        result = await orchestrator.process(context, stream_callback)
        
        # Display results
        print("\n" + "="*50)
        print("FINAL ANSWER:")
        print("="*50)
        print(result.synthesized_answer if result.synthesized_answer else "No answer generated")
        
        if result.provenance_map:
            print("\n" + "="*50)
            print("PROVENANCE MAP (First 3 entries):")
            print("="*50)
            for link in result.provenance_map[:3]:
                print(f"\nSentence: {link.sentence[:100]}...")
                print(f"Source Agent: {link.source_agent.value}")
                if link.source_chunks:
                    print(f"Source Chunks: {', '.join(link.source_chunks[:2])}")
        
        print("\n" + "="*50)
        print(f"Status: {result.status}")
        print(f"Total claims: {len(result.claims)}")
        print(f"Total critiques: {len(result.critiques)}")
        print(f"Total policy violations: {len(result.policy_violations)}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())