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
        self.use_fallback = os.getenv("LLM_FALLBACK", "false").lower() in ("true", "1", "yes")

        if not self.api_key:
            logger.warning(
                "LLM fallback enabled: GROQ_API_KEY not found. "
                "Using local fallback generator for offline and rate-limited execution."
            )
            self.use_fallback = True
        else:
            logger.info("Groq API key found")

        if not self.use_fallback:
            try:
                self.client = AsyncGroq(api_key=self.api_key)
            except Exception as exc:
                logger.warning("groq_client_init_failed", extra={"error": str(exc)})
                self.client = None
                self.use_fallback = True
        else:
            self.client = None

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0

        logger.info(f"LLM Client Ready: {self.model} (fallback={self.use_fallback})")

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

        try:
            if self.use_fallback or not self.client:
                return self._fallback_generate(prompt, temperature, max_tokens)

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
                        self.use_fallback = True
                        fallback = self._fallback_generate(prompt, temperature, max_tokens)
                        fallback["error"] = str(e)
                        fallback["fallback"] = True
                        fallback["attempts"] = attempt
                        return fallback
                    await self._wait_for_retry(attempt, e)

            fallback = self._fallback_generate(prompt, temperature, max_tokens)
            fallback["error"] = str(last_error) if last_error else "Unknown error"
            fallback["fallback"] = True
            fallback["attempts"] = self.max_retries + 1
            return fallback
        except Exception as exc:
            logger.exception("LLM generation terminal failure")
            self.use_fallback = True
            fallback = self._fallback_generate(prompt, temperature, max_tokens)
            fallback["error"] = str(exc)
            fallback["fallback"] = True
            fallback["attempts"] = 0
            return fallback

    def get_stats(self):
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": self.total_cost,
            "model": self.model,
            "provider": "Groq (Free)" if not self.use_fallback else "local_fallback"
        }

    def _fallback_generate(self, prompt: str, temperature: float, max_tokens: int):
        import json
        import re

        def safe_json(payload):
            return json.dumps(payload)

        prompt_lower = prompt.lower()
        text = ""

        if "break this query into sub-tasks" in prompt_lower or "sub-tasks" in prompt_lower:
            tasks = [
                {"id": 1, "description": "Retrieve relevant information for the query.", "task_type": "retrieval", "depends_on": []},
                {"id": 2, "description": "Summarize retrieved evidence and create an answer.", "task_type": "synthesis", "depends_on": [1]},
            ]
            text = safe_json({"tasks": tasks})
        elif "extract grounded claims" in prompt_lower or "grounded claims" in prompt_lower:
            chunk_ids = re.findall(r"chunk_id[:=]?(\w+)", prompt_lower)
            chunk_ids = chunk_ids or ["chroma_0"]
            claims = [
                {"text": "The retrieved sources describe the requested concept.", "chunk_ids": [chunk_ids[0]], "confidence": 0.75}
            ]
            text = safe_json({"claims": claims})
        elif "review this claim" in prompt_lower or "check for:" in prompt_lower:
            text = safe_json({"has_issue": False})
        elif "follow_up_queries" in prompt_lower or "propose focused follow-up" in prompt_lower:
            query_match = re.search(r"user question:\s*(.*?)\n", prompt, re.IGNORECASE)
            base = query_match.group(1) if query_match else "the query"
            text = safe_json({"entities": [], "follow_up_queries": [f"More details about {base}"[:80]]})
        elif "final answer:" in prompt_lower or "synthesize a final answer" in prompt_lower:
            claims = re.findall(r"- (.*?)(?:\\n|$)", prompt)
            if not claims:
                claims = ["Information from retrieved snippets is summarized below."]
            answer = " ".join(c.strip() for c in claims[:3])
            if not answer:
                answer = "No grounded answer could be generated from the current evidence."
            text = answer
        else:
            text = ""

        return {
            "text": text,
            "input_tokens": self.count_tokens(prompt),
            "output_tokens": self.count_tokens(text),
            "total_tokens": self.count_tokens(prompt) + self.count_tokens(text),
            "cost": 0,
            "model": self.model,
            "attempts": 1,
        }
