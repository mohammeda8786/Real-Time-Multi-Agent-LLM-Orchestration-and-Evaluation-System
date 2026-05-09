import uuid
from typing import Optional, Callable, Awaitable, List, Dict, Any
from app.agents.base import BaseAgent
from app.models.schemas import (
    SharedContext, RetrievedChunk, Claim, AgentType
)
import logging

logger = logging.getLogger(__name__)

class RAGAgent(BaseAgent):
    """
    Multi-hop retrieval agent that performs at least two retrieval steps
    before forming an answer. Each chunk's contribution is cited.
    """
    
    def __init__(self):
        super().__init__(AgentType.RAG)
        self.max_hops = 3  # Minimum 2 hops required
    
    async def process(
        self,
        context: SharedContext,
        streaming_callback: Optional[Callable[[str, dict], Awaitable[None]]] = None
    ) -> SharedContext:
        
        await self._emit_event(streaming_callback, "rag_start",
                              message="Starting multi-hop retrieval")
        
        # Step 1: Initial retrieval based on query
        hop1_results = await self._retrieval_hop(
            query=context.original_query,
            hop_number=1,
            context=context
        )
        
        # Step 2: Extract key concepts from first hop for second hop
        key_concepts = await self._extract_key_concepts(hop1_results)
        
        hop2_results = await self._retrieval_hop(
            query=f"Based on: {context.original_query}\nKey concepts: {key_concepts}",
            hop_number=2,
            context=context
        )
        
        # Step 3: Optional third hop for depth (if needed)
        hop3_results = []
        if await self._needs_third_hop(hop1_results, hop2_results, context.original_query):
            refined_query = await self._refine_query(hop1_results, hop2_results, context.original_query)
            hop3_results = await self._retrieval_hop(
                query=refined_query,
                hop_number=3,
                context=context
            )
        
        # Step 4: Combine all retrieved chunks
        all_chunks = hop1_results + hop2_results + hop3_results
        context.retrieved_chunks.extend(all_chunks)
        
        # Step 5: Generate claims with chunk citations
        claims = await self._generate_claims_with_citations(
            all_chunks,
            context.original_query
        )
        context.claims.extend(claims)
        
        # Step 6: Record multi-hop reasoning path
        context.multi_hop_reasoning_path = [
            {"hop": 1, "chunks": [c.chunk_id for c in hop1_results]},
            {"hop": 2, "chunks": [c.chunk_id for c in hop2_results]},
            {"hop": 3, "chunks": [c.chunk_id for c in hop3_results]} if hop3_results else {}
        ]
        
        await self._emit_event(streaming_callback, "rag_complete",
                              chunks_retrieved=len(all_chunks),
                              hops_completed=2 if not hop3_results else 3,
                              claims_generated=len(claims))
        
        return context
    
    async def _retrieval_hop(self, query: str, hop_number: int, context: SharedContext) -> List[RetrievedChunk]:
        """Single retrieval hop with source citation"""
        
        await self._emit_event(None, "retrieval_hop", hop=hop_number, query=query[:100])
        
        # In production: actual vector DB or web search
        # For demo, mock retrieval with diverse sources
        
        mock_chunks = [
            RetrievedChunk(
                chunk_id=f"chunk_{hop_number}_1",
                content=f"Source A information about: {query[:50]}... (Hop {hop_number} retrieval)",
                source=f"https://source-a.com/hop{hop_number}",
                relevance_score=0.85,
                retrieval_step=hop_number
            ),
            RetrievedChunk(
                chunk_id=f"chunk_{hop_number}_2",
                content=f"Source B details on: {query[:50]}... (Hop {hop_number} retrieval)",
                source=f"https://source-b.com/hop{hop_number}",
                relevance_score=0.78,
                retrieval_step=hop_number
            ),
            RetrievedChunk(
                chunk_id=f"chunk_{hop_number}_3",
                content=f"Source C perspective about: {query[:50]}... (Hop {hop_number} retrieval)",
                source=f"https://source-c.com/hop{hop_number}",
                relevance_score=0.72,
                retrieval_step=hop_number
            )
        ]
        
        return mock_chunks
    
    async def _extract_key_concepts(self, chunks: List[RetrievedChunk]) -> str:
        """Extract key concepts from retrieved chunks for next hop"""
        # In production: use LLM to extract entities/concepts
        concepts = []
        for chunk in chunks:
            # Simple keyword extraction for demo
            words = chunk.content.split()[:10]
            concepts.extend(words)
        
        return " ".join(concepts[:20])
    
    async def _needs_third_hop(self, hop1: List[RetrievedChunk], 
                               hop2: List[RetrievedChunk], 
                               query: str) -> bool:
        """Determine if third hop is needed for completeness"""
        # Check if we have sufficient information
        avg_relevance = sum(c.relevance_score for c in hop2) / len(hop2) if hop2 else 0
        
        # Third hop if relevance is low or query is complex
        return avg_relevance < 0.7 or len(query.split()) > 15
    
    async def _refine_query(self, hop1: List[RetrievedChunk], 
                            hop2: List[RetrievedChunk], 
                            original_query: str) -> str:
        """Refine query for third hop based on previous results"""
        # Combine insights from previous hops
        combined = f"{original_query}\nMissing information: "
        
        # Identify what's missing (simplified)
        if hop1 and hop2:
            combined += "Need more recent information"
        
        return combined
    
    async def _generate_claims_with_citations(self, 
                                              chunks: List[RetrievedChunk], 
                                              query: str) -> List[Claim]:
        """Generate claims where each claim cites which chunks support it"""
        
        # In production: LLM that outputs claims with citations
        # For demo, create mock claims
        
        claims = []
        
        # Group chunks by topic (simplified)
        for idx, chunk in enumerate(chunks):
            claim = Claim(
                claim_id=str(uuid.uuid4()),
                text=f"According to {chunk.source}, {chunk.content[:100]}...",
                agent_source=self.agent_type,
                chunk_citations=[chunk.chunk_id],
                confidence_score=chunk.relevance_score,
                start_char=0,
                end_char=100
            )
            claims.append(claim)
        
        # Add synthesized claim from multiple chunks
        if len(chunks) >= 2:
            multi_chunk_claim = Claim(
                claim_id=str(uuid.uuid4()),
                text=f"Combining information from multiple sources, we find that {query[:50]}...",
                agent_source=self.agent_type,
                chunk_citations=[c.chunk_id for c in chunks[:3]],
                confidence_score=0.82,
                start_char=0,
                end_char=100
            )
            claims.append(multi_chunk_claim)
        
        return claims