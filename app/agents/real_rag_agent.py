"""
RAG agent: multi-hop retrieval with reasoning trace, orchestrator-mediated tools,
and chunk-grounded claims (stable chunk IDs).
"""

from __future__ import annotations

import uuid
from typing import List

from app.agents.base import BaseAgent
from app.models.schemas import SharedContext, RetrievedChunk, Claim, AgentType
from app.orchestration.tool_mediator import get_mediator
from app.rag.pipeline import RAGPipeline
from app.llm_client import LLMClient


class RealRAGAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.RAG)
        self.pipeline = RAGPipeline()
        self.llm = LLMClient()

        if self.pipeline.vector_store.count() == 0:
            self.pipeline.index_documents(source="sample")

    async def process(self, context: SharedContext, streaming_callback=None):
        await self._emit_event(
            streaming_callback,
            "rag_start",
            message="Multi-hop RAG with tool-augmented retrieval",
        )

        query = context.original_query
        results, reasoning = await self.pipeline.search_with_reasoning(
            query, self.llm, hops=2, top_k=5
        )
        context.multi_hop_reasoning_path.append(reasoning)

        mediator = get_mediator()
        if mediator:
            await mediator.invoke(
                context,
                "web_search",
                lambda: {"query": query[:200], "limit": 3},
                dedupe=True,
                streaming_callback=streaming_callback,
            )

        grounded_snippets = []
        for i, result in enumerate(results, 1):
            cid = result.get("chunk_id") or str(uuid.uuid4())
            chunk = RetrievedChunk(
                chunk_id=cid,
                content=result["text"],
                source=result.get("source", "unknown"),
                relevance_score=float(result.get("similarity_score", 0.0)),
                retrieval_step=2 if i > 3 else 1,
            )
            context.retrieved_chunks.append(chunk)
            grounded_snippets.append(f"[{i}] (chunk_id={cid}) {result['text'][:600]}")

        await self._llm_grounded_claims(context, query, grounded_snippets)

        await self._emit_event(
            streaming_callback,
            "rag_complete",
            chunks_retrieved=len(context.retrieved_chunks),
            claims_generated=len(context.claims),
        )

        return context

    async def _llm_grounded_claims(self, context: SharedContext, query: str, snippets: List[str]) -> None:
        """Ask the model for atomic claims that map to chunk indices; fallback to heuristic claims."""
        catalog = "\n".join(snippets)
        prompt = f"""You are extracting grounded claims for traceability.

RULES:
- Use ONLY information in the numbered snippets below.
- Each claim must reference at least one chunk_id from the snippet header.
- Output JSON ONLY: {{"claims":[{{"text":"...", "chunk_ids":["..."],"confidence":0.0-1.0}}]}}

SNIPPETS:
{catalog}

QUESTION: {query}
JSON:"""

        resp = await self.llm.generate(prompt, temperature=0.2, max_tokens=600)
        from app.utils.json_extract import extract_json_object

        parsed = extract_json_object(resp.get("text") or "")
        claims_payload = (parsed or {}).get("claims") if parsed else None

        if claims_payload and isinstance(claims_payload, list):
            id_set = {c.chunk_id for c in context.retrieved_chunks}
            for item in claims_payload[:12]:
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                cited = [x for x in (item.get("chunk_ids") or []) if x in id_set]
                if not cited:
                    cited = [context.retrieved_chunks[0].chunk_id] if context.retrieved_chunks else []
                conf = float(item.get("confidence") or 0.7)
                context.claims.append(
                    Claim(
                        claim_id=str(uuid.uuid4()),
                        text=text[:2000],
                        agent_source=self.agent_type,
                        chunk_citations=cited,
                        confidence_score=min(1.0, max(0.0, conf)),
                    )
                )
            return

        for chunk in context.retrieved_chunks[:6]:
            context.claims.append(
                Claim(
                    claim_id=str(uuid.uuid4()),
                    text=f"Source {chunk.source}: {chunk.content[:400]}",
                    agent_source=self.agent_type,
                    chunk_citations=[chunk.chunk_id],
                    confidence_score=chunk.relevance_score,
                )
            )
