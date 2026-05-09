"""
Groq LLM Client - Free & Fast
"""

from groq import AsyncGroq
import os
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self, model: str = None):
        self.model = model or os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
        self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            print("❌ GROQ_API_KEY not found in .env file")
            print("   Get free key from: https://console.groq.com/keys")
        else:
            print(f"✅ Groq API key found")
        
        self.client = AsyncGroq(api_key=self.api_key)
        
        # Token tracking (approximate for Groq)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0  # Groq is FREE!
        
        print(f"✅ Groq LLM Client Ready: {self.model}")
        print(f"   (Free API - no cost)")
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count (Groq doesn't provide exact count)"""
        if not text:
            return 0
        # Rough estimate: ~4 chars per token
        return len(text) // 4
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500):
        """Generate response using Groq (FREE)"""
        
        input_tokens = self.count_tokens(prompt)
        self.total_input_tokens += input_tokens
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            output_text = response.choices[0].message.content
            output_tokens = self.count_tokens(output_text)
            
            self.total_output_tokens += output_tokens
            
            return {
                "text": output_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost": 0,  # Groq is FREE!
                "model": self.model
            }
        except Exception as e:
            print(f"❌ Groq Error: {e}")
            return {
                "text": f"Error: {e}",
                "error": str(e),
                "total_tokens": 0,
                "cost": 0
            }
    
    def get_stats(self):
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": 0,
            "model": self.model,
            "provider": "Groq (Free)"
        }