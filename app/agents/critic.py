"""
Critic Agent - Uses LLM to review each claim individually
"""

from app.agents.base import BaseAgent
from app.models.schemas import SharedContext, Critique, Claim, AgentType
from app.llm_client import LLMClient
import uuid
from typing import List

class CriticAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.CRITIC)
        self.llm = LLMClient()
    
    async def process(self, context: SharedContext, streaming_callback=None):
        await self._emit_event(streaming_callback, "critic_start",
                              message=f"LLM reviewing {len(context.claims)} claims")
        
        critiqued_claim_ids = {c.claim_id for c in context.critiques}
        uncritiqued_claims = [c for c in context.claims if c.claim_id not in critiqued_claim_ids]
        
        for claim in uncritiqued_claims:
            critique = await self._llm_review_claim(claim, context.claims)
            if critique:
                context.critiques.append(critique)
                await self._emit_event(streaming_callback, "critique_generated",
                                      claim_id=claim.claim_id,
                                      disagreement=critique.disagree_text[:100])
        
        await self._emit_event(streaming_callback, "critic_complete",
                              critiques_generated=len(context.critiques))
        
        return context
    
    async def _llm_review_claim(self, claim: Claim, all_claims: List[Claim]):
        """Use LLM to review a single claim"""
        
        all_claims_text = "\n".join([f"- {c.text[:100]}..." for c in all_claims if c.claim_id != claim.claim_id])
        
        prompt = f"""Review this claim and find issues:

CLAIM: "{claim.text}"

OTHER CLAIMS:
{all_claims_text}

Check for:
1. Contradictions with other claims
2. Missing citations
3. Factual errors
4. Vague language

If you find an issue, return JSON:
{{"has_issue": true, "disagree_text": "specific text from claim", "reason": "why it's wrong", "suggestion": "how to fix"}}

If no issues, return: {{"has_issue": false}}
"""
        
        response = await self.llm.generate(prompt, temperature=0.3, max_tokens=200)
        
        try:
            import json
            result = json.loads(response["text"])
            if result.get("has_issue"):
                return Critique(
                    claim_id=claim.claim_id,
                    disagree_text=result["disagree_text"],
                    disagreement_reason=result["reason"],
                    suggested_correction=result.get("suggestion"),
                    confidence_in_critique=0.8
                )
        except:
            pass
        
        return None