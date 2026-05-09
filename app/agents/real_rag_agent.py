"""
Real RAG Agent - Uses LLM to generate answers from retrieved chunks
"""

from app.agents.base import BaseAgent
from app.models.schemas import SharedContext, RetrievedChunk, Claim, AgentType
from app.rag.pipeline import RAGPipeline
from app.llm_client import LLMClient
from typing import List
import uuid

class RealRAGAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.RAG)
        self.pipeline = RAGPipeline()
        self.llm = LLMClient()  # ← LLM for answer generation
        
        if self.pipeline.vector_store.count() == 0:
            self.pipeline.index_documents(source="sample")
        
        print(f"✅ Real RAG Agent with LLM")
    
    async def process(self, context: SharedContext, streaming_callback=None):
        await self._emit_event(streaming_callback, "rag_start",
                              message="LLM-powered RAG: Retrieving and generating")
        
        query = context.original_query
        
        # Step 1: Retrieve relevant chunks
        results = self.pipeline.search(query, hops=2, top_k=3)
        
        # Step 2: Use LLM to generate answer with citations
        answer = await self._llm_generate_answer(results, query)
        
        # Step 3: Create claims from results
        for result in results:
            chunk_id = str(uuid.uuid4())
            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                content=result["text"],
                source=result["source"],
                relevance_score=result["similarity_score"],
                retrieval_step=1
            )
            context.retrieved_chunks.append(chunk)
            
            claim = Claim(
                claim_id=str(uuid.uuid4()),
                text=f"According to {result['source']}: {result['text'][:150]}...",
                agent_source=self.agent_type,
                chunk_citations=[chunk_id],
                confidence_score=result["similarity_score"]
            )
            context.claims.append(claim)
        
        await self._emit_event(streaming_callback, "rag_complete",
                              chunks=len(context.retrieved_chunks),
                              claims=len(context.claims))
        
        return context
    
    async def _llm_generate_answer(self, chunks: List[dict], query: str) -> str:
        """Use LLM to generate answer from retrieved chunks"""
        
        # Build context from chunks
        context_text = ""
        for i, chunk in enumerate(chunks, 1):
            context_text += f"[{i}] Source: {chunk['source']}\n{chunk['text']}\n\n"
        
        prompt = f"""Based on these sources, answer the question. Cite sources using [1], [2], etc.

SOURCES:
{context_text}

QUESTION: {query}

ANSWER:"""
        
        response = await self.llm.generate(prompt, temperature=0.5, max_tokens=500)
        return response["text"]