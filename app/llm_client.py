"""
Groq LLM Client - Free & Fast
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model: str = None):
        self.model = model or os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
        self.api_key = os.getenv("GROQ_API_KEY")
        self.max_retries = int(os.getenv("LLM_RETRY_MAX", "3"))
        self.base_backoff = float(os.getenv("LLM_RETRY_BACKOFF", "1.0"))

        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in .env file. LLM calls may fail.")
        else:
            logger.info("Groq API key found")

        self.client = AsyncGroq(api_key=self.api_key)

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0

        logger.info(f"Groq LLM Client Ready: {self.model}")

    def count_tokens(self, text: str) -> int:
        """Approximate token count (Groq doesn't provide exact count)"""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _should_retry(self, error: Exception) -> bool:
        message = str(error).lower()
        return any(keyword in message for keyword in ["rate limit", "429", "too many requests", "retry-after"])

    async def _wait_for_retry(self, attempt: int, error: Exception):
        delay = self.base_backoff * (2 ** (attempt - 1))
        logger.warning(
            "LLM request failed on attempt %s: %s. Backing off for %.1fs...",
            attempt, error, delay
        )
        await asyncio.sleep(delay)

    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500):
        """Generate response using Groq with retry and backoff."""

        input_tokens = self.count_tokens(prompt)
        self.total_input_tokens += input_tokens

        last_error = None
        for attempt in range(1, self.max_retries + 2):
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
                    "cost": 0,
                    "model": self.model,
                    "attempts": attempt
                }
            except Exception as e:
                last_error = e
                if attempt > self.max_retries or not self._should_retry(e):
                    logger.error("LLM generation failed: %s", e)
                    return {
                        "text": "",
                        "error": str(e),
                        "total_tokens": 0,
                        "cost": 0,
                        "model": self.model,
                        "attempts": attempt
                    }
                await self._wait_for_retry(attempt, e)

        return {
            "text": "",
            "error": str(last_error) if last_error else "Unknown error",
            "total_tokens": 0,
            "cost": 0,
            "model": self.model,
            "attempts": self.max_retries + 1
        }

    def get_stats(self):
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "model": self.model,
            "provider": "Groq (Free)"
        }
