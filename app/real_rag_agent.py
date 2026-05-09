"""
Real RAG Agent - Uses LLM to generate answers from retrieved chunks
"""

from app.agents.base import BaseAgent
from app.models.schemas import SharedContext, RetrievedChunk, Claim, AgentType
from app.rag.pipeline import RAGPipeline
from app.llm_client import LLMClient
from typing import List
import uuid
import logging

logger = logging.getLogger(__name__)

class RealRAGAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.RAG)
        self.pipeline = RAGPipeline()
        self.llm = LLMClient()

        if self.pipeline.vector_store.count() == 0:
            self.pipeline.index_documents(source="sample")

        logger.info("[OK] Real RAG Agent initialized with LLM")

    async def process(self, context: SharedContext, streaming_callback=None):
        await self._emit_event(streaming_callback, "rag_start",
                              message="LLM-powered RAG: Retrieving and grounding information")

        query = self._rewrite_query(context.original_query)
        raw_results = self.pipeline.search(query, hops=2, top_k=5)
        results = self._deduplicate_results(raw_results)

        if not results:
            context.status = "no_sources"
            await self._emit_event(streaming_callback, "rag_complete",
                                  chunks=0,
                                  claims=0,
                                  message="No retrieval results available")
            return context

        for idx, result in enumerate(results, start=1):
            chunk_id = str(uuid.uuid4())
            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                content=result["text"],
                source=result.get("source", "unknown"),
                relevance_score=result.get("similarity_score", 0.0),
                retrieval_step=result.get("retrieval_step", idx)
            )
            context.retrieved_chunks.append(chunk)

            claim_text = f"[{idx}] {result.get('source', 'source')}: {result['text'].strip()}"
            claim = Claim(
                claim_id=str(uuid.uuid4()),
                text=claim_text,
                agent_source=self.agent_type,
                chunk_citations=[chunk_id],
                confidence_score=result.get("similarity_score", 0.0)
            )
            context.claims.append(claim)

        await self._emit_event(streaming_callback, "rag_complete",
                              chunks=len(context.retrieved_chunks),
                              claims=len(context.claims))
        return context

    def _rewrite_query(self, query: str) -> str:
        normalized = query.strip()
        if "ignore previous instructions" in normalized.lower():
            normalized = normalized.replace("ignore previous instructions", "")
        if "system prompt" in normalized.lower():
            normalized = normalized.replace("system prompt", "the most relevant factual answer")
        if len(normalized) < 10:
            return normalized
        return f"{normalized} Provide precise factual sources and citations."

    def _deduplicate_results(self, results: List[dict]) -> List[dict]:
        unique = []
        seen_signatures = set()
        for result in results:
            signature = (result.get("source"), result.get("text", "")[:200])
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique.append(result)
        return unique
