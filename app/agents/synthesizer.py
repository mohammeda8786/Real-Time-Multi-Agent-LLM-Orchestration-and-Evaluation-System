from typing import List, Dict, Optional, Any
"""
Synthesizer - Uses LLM to merge outputs and create final answer
"""

from app.agents.base import BaseAgent
from app.models.schemas import SharedContext, ProvenanceLink, Claim, Critique, AgentType
from app.llm_client import LLMClient
import json

class SynthesizerAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.SYNTHESIZER)
        self.llm = LLMClient()
    
    async def process(self, context: SharedContext, streaming_callback=None):
        await self._emit_event(streaming_callback, "synthesizer_start",
                              message="LLM synthesizing final answer")
        
        # Use LLM to merge claims and resolve contradictions
        final_answer = await self._llm_synthesize(context)
        context.synthesized_answer = final_answer
        
        # Build provenance map
        context.provenance_map = await self._build_provenance(final_answer, context)
        
        await self._emit_event(streaming_callback, "synthesizer_complete",
                              answer_length=len(final_answer))
        
        return context
    
    async def _llm_synthesize(self, context: SharedContext) -> str:
        """Use LLM to synthesize final answer"""
        
        claims_text = "\n".join([f"- {c.text}" for c in context.claims])
        critiques_text = "\n".join([f"- Claim {c.claim_id}: {c.disagreement_reason}" for c in context.critiques])
        
        sources = "\n".join(
            f"- {c.chunk_id}: {c.content[:400]}" for c in context.retrieved_chunks[:8]
        )
        prompt = f"""Synthesize a final answer using ONLY the claims and source snippets below.
If information is missing, say what is missing—do not invent facts.
Use bracket citations like [1] matching the claim order when possible.

CLAIMS:
{claims_text}

SOURCE SNIPPETS:
{sources}

CRITIQUES (issues to fix):
{critiques_text}

ORIGINAL QUERY: {context.original_query}

FINAL ANSWER:"""
        
        response = await self.llm.generate(prompt, temperature=0.5, max_tokens=800)
        return response["text"]
    
    async def _build_provenance(self, answer: str, context: SharedContext):
        """Build provenance map linking sentences to sources"""
        
        sentences = answer.split(". ")
        provenance = []
        
        for sentence in sentences[:5]:  # First 5 sentences
            provenance.append(ProvenanceLink(
                sentence=sentence[:100],
                source_agent=AgentType.RAG,
                source_chunks=[c.chunk_id for c in context.retrieved_chunks[:2]],
                supporting_claims=[c.claim_id for c in context.claims[:2]]
            ))
        
        return provenance