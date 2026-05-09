# simple_llm_test.py - No imports needed
import asyncio

async def test():
    print("="*50)
    print("SIMPLE LLM TEST")
    print("="*50)
    
    # First check if openai is installed
    try:
        import openai
        print("✅ OpenAI installed")
    except ImportError:
        print("❌ OpenAI not installed. Run: pip install openai")
        return
    
    # Check if tiktoken is installed
    try:
        import tiktoken
        print("✅ TikToken installed")
    except ImportError:
        print("❌ TikToken not installed. Run: pip install tiktoken")
        return
    
    # Now try to import your LLM client
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from app.llm_client import LLMClient
        print("✅ LLMClient imported successfully")
        
        llm = LLMClient()
        print("✅ LLM Client created")
        
        # Test token counting
        tokens = llm.count_tokens("Hello world")
        print(f"✅ Token counting: 'Hello world' = {tokens} tokens")
        
        # Test generation (requires API key)
        print("\n🔄 Testing generation...")
        result = await llm.generate("Say 'hello' in one word")
        print(f"✅ Response: {result['text']}")
        print(f"📊 Tokens used: {result['tokens']}")
        
        print("\n🎉 LLM IS WORKING PERFECTLY!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
