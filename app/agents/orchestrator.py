from typing import List, Dict, Optional, Any
"""
Orchestrator - Uses LLM to decide which agent runs next
"""

from app.agents.base import BaseAgent
from app.agents.decomposer import DecompositionAgent
from app.agents.real_rag_agent import RealRAGAgent
from app.agents.critic import CriticAgent
from app.agents.synthesizer import SynthesizerAgent
from app.models.schemas import SharedContext, AgentType, RoutingDecision, ContextBudget
from app.llm_client import LLMClient
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.ORCHESTRATOR)
        self.sub_agents = {
            AgentType.DECOMPOSER: DecompositionAgent(),
            AgentType.RAG: RealRAGAgent(),
            AgentType.CRITIC: CriticAgent(),
            AgentType.SYNTHESIZER: SynthesizerAgent()
        }
        self.completed_agents = set()
        self.last_agent_run = None
        self.llm = LLMClient()  # ← LLM for routing
        print("✅ Orchestrator with LLM routing")
    
    async def process(self, context: SharedContext, streaming_callback=None):
        await self._emit_event(streaming_callback, "orchestrator_start", 
                              message="Starting LLM-powered dynamic routing")
        
        await self._initialize_budgets(context)
        
        max_iterations = 10
        iteration = 0
        agent_run_count = {}
        
        while iteration < max_iterations:
            iteration += 1
            
            # Use LLM to decide next agent
            routing_decision = await self._llm_decide_next_agent(context)
            
            if not routing_decision:
                break
            
            agent_name = routing_decision.next_agent.value
            agent_run_count[agent_name] = agent_run_count.get(agent_name, 0) + 1
            
            if agent_run_count[agent_name] > 2:
                break
            
            await self._emit_event(streaming_callback, "routing_decision",
                                  agent=routing_decision.next_agent.value,
                                  reasoning=routing_decision.reasoning)
            
            if routing_decision.next_agent in self.sub_agents:
                agent = self.sub_agents[routing_decision.next_agent]
                context = await agent.process(context, streaming_callback)
                self.completed_agents.add(routing_decision.next_agent)
                self.last_agent_run = routing_decision.next_agent
        
        # Final synthesis
        if not context.synthesized_answer:
            synthesizer = self.sub_agents[AgentType.SYNTHESIZER]
            context = await synthesizer.process(context, streaming_callback)
        
        context.status = "completed"
        context.completed_at = datetime.now()
        
        # Show LLM usage stats
        stats = self.llm.get_stats()
        await self._emit_event(streaming_callback, "llm_stats", 
                              total_tokens=stats["total_tokens"],
                              total_cost=stats["total_cost"])
        
        return context
    
    async def _llm_decide_next_agent(self, context: SharedContext):
        """Use LLM to decide which agent runs next"""
        
        prompt = f"""You are a multi-agent orchestrator. Current state:

Query: "{context.original_query}"
Has decomposition: {context.dependency_graph is not None}
Retrieved chunks: {len(context.retrieved_chunks)}
Claims: {len(context.claims)}
Critiques: {len(context.critiques)}
Has answer: {context.synthesized_answer is not None}

Available agents:
- DECOMPOSER: Break down complex queries into tasks
- RAG: Retrieve information from knowledge base
- CRITIC: Review claims for contradictions
- SYNTHESIZER: Create final answer

Which agent should run NEXT? Return ONLY the agent name.
If all work is done, return "DONE".
"""
        
        response = await self.llm.generate(prompt, temperature=0.2, max_tokens=20)
        decision = response["text"].strip().upper()
        
        agent_map = {
            "DECOMPOSER": AgentType.DECOMPOSER,
            "RAG": AgentType.RAG,
            "CRITIC": AgentType.CRITIC,
            "SYNTHESIZER": AgentType.SYNTHESIZER
        }
        
        if decision in agent_map:
            return RoutingDecision(
                next_agent=agent_map[decision],
                reasoning=f"LLM decided: {decision}",
                priority=9,
                context_budget_allocation=2000,
                expected_output_type="varies"
            )
        return None
    
    async def _initialize_budgets(self, context: SharedContext):
        base_budget = 8000
        budget_map = {
            AgentType.DECOMPOSER: 1200,
            AgentType.RAG: 2400,
            AgentType.CRITIC: 1600,
            AgentType.SYNTHESIZER: 2800,
        }
        for agent_type, allocated in budget_map.items():
            context.budgets[agent_type] = ContextBudget(
                agent_type=agent_type,
                allocated_tokens=allocated,
                used_tokens=0,
                remaining_tokens=allocated
            )
    
    async def _allocate_budget(self, context: SharedContext, agent_type: AgentType, tokens: int):
        if agent_type in context.budgets:
            budget = context.budgets[agent_type]
            budget.allocated_tokens = tokens
            budget.remaining_tokens = tokens - budget.used_tokens