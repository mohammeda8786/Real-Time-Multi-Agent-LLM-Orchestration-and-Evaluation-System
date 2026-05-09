import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def main():
    print("="*50)
    print("TESTING GROQ LLM (FREE)")
    print("="*50)
    
    from app.llm_client import LLMClient
    
    # Create LLM client
    llm = LLMClient()
    
    # Test 1: Token counting
    print("\n1. Testing token counting...")
    text = "Hello, how are you?"
    tokens = llm.count_tokens(text)
    print(f"   '{text}' = {tokens} tokens (approx)")
    
    # Test 2: Generate response
    print("\n2. Testing generation (FREE)...")
    result = await llm.generate("What is Python? Answer in one short sentence.")
    print(f"   Response: {result['text']}")
    print(f"   Tokens used: {result['total_tokens']}")
    print(f"   Cost: ${result['cost']} (FREE!)")
    
    # Test 3: More complex query
    print("\n3. Testing complex query...")
    result2 = await llm.generate("Explain machine learning in 2 sentences.")
    print(f"   Response: {result2['text']}")
    
    # Test 4: Stats
    print("\n4. Usage Stats:")
    stats = llm.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ GROQ LLM IS WORKING PERFECTLY (FREE)!")

if __name__ == "__main__":
    asyncio.run(main())