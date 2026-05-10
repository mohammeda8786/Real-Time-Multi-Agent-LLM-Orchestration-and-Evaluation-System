from typing import List, Dict, Optional, Any
"""
Orchestrator - Uses LLM to decide which agent runs next
"""

from app.agents.base import BaseAgent
from app.agents.decomposer import DecompositionAgent
from app.agents.real_rag_agent import RealRAGAgent
from app.agents.critic import CriticAgent
from app.agents.synthesizer import SynthesizerAgent
from app.context.budget_manager import ContextBudgetManager
from app.models.schemas import SharedContext, AgentType, RoutingDecision, ContextBudget
from app.llm_client import LLMClient
from app.security.query_guard import assess_query, QueryRisk
from app.orchestration.tool_mediator import ToolMediator
from datetime import datetime
import logging
import asyncio
import math

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
        self.budget_manager = ContextBudgetManager()
        logger.info("[OK] Orchestrator initialized with LLM routing")

    def is_ready(self) -> bool:
        try:
            rag_agent = self.sub_agents.get(AgentType.RAG)
            vector_ready = rag_agent and rag_agent.pipeline.vector_store.count() >= 0
            return self.llm is not None and bool(vector_ready)
        except Exception:
            return False
    
    async def process(self, context: SharedContext, streaming_callback=None):
        await self._emit_event(streaming_callback, "orchestrator_start", 
                              message="Starting LLM-powered dynamic routing")

        guard = assess_query(context.original_query)
        if guard.risk != QueryRisk.NONE:
            context.policy_violations.append({
                "type": "query_guard",
                "risk": guard.risk.value,
                "reason": guard.reason,
                "timestamp": datetime.now().isoformat(),
            })
            context.synthesized_answer = guard.safe_response or "Unable to process this query safely."
            context.status = "completed"
            context.completed_at = datetime.now()
            await self._emit_event(streaming_callback, "orchestrator_early_exit", risk=guard.risk.value)
            return context

        mediator = ToolMediator(job_id=context.job_id)
        med_token = mediator.attach()
        try:
            await self.budget_manager.initialize_budgets(context)
            context = await self._run_agents_loop(context, streaming_callback)
        finally:
            ToolMediator.detach(med_token)

        context.status = "completed"
        context.completed_at = datetime.now()

        await self._stream_answer_chunks(context, streaming_callback)

        stats = self.llm.get_stats()
        await self._emit_event(streaming_callback, "llm_stats", 
                              total_tokens=stats["total_tokens"],
                              total_cost=stats["total_cost"])

        return context

    async def _run_agents_loop(self, context: SharedContext, streaming_callback):
        max_iterations = 10
        iteration = 0
        agent_run_count = {}

        while iteration < max_iterations:
            iteration += 1

            routing_decision = await self._llm_decide_next_agent(context)
            if not routing_decision:
                logger.info("Orchestrator received DONE or no route; ending loop")
                break

            agent_name = routing_decision.next_agent.value
            agent_run_count[agent_name] = agent_run_count.get(agent_name, 0) + 1

            if agent_run_count[agent_name] > 2:
                logger.warning("Agent %s selected too often, breaking loop", agent_name)
                break

            context.routing_decisions.append(routing_decision)
            t0 = asyncio.get_event_loop().time()
            await self._emit_event(streaming_callback, "routing_decision",
                                  agent=agent_name,
                                  reasoning=routing_decision.reasoning,
                                  trace_id=context.trace_id,
                                  job_id=context.job_id)

            if routing_decision.next_agent in self.sub_agents:
                agent = self.sub_agents[routing_decision.next_agent]
                
                try:
                    context = await agent.process(context, streaming_callback)
                    latency_ms = (asyncio.get_event_loop().time() - t0) * 1000
                    await self._log_execution(context, "agent_run", {
                        "agent": agent_name,
                        "iteration": iteration,
                        "status": context.status,
                        "claimed_outputs": len(context.claims),
                        "latency_ms": latency_ms,
                        "trace_id": context.trace_id,
                    })
                    self.completed_agents.add(routing_decision.next_agent)
                    self.last_agent_run = routing_decision.next_agent

                    budget_used = routing_decision.context_budget_allocation or 0
                    budget_ok = await self._consume_budget(context, budget_used, f"agent_{agent_name}_execution")
                    remaining = context.budgets.get(routing_decision.next_agent).remaining_tokens if routing_decision.next_agent in context.budgets else None
                    await self._emit_event(
                        streaming_callback,
                        "budget_update",
                        agent=agent_name,
                        allocated=budget_used,
                        remaining_tokens=remaining,
                        budget_ok=budget_ok,
                    )
                    
                except Exception as e:
                    # Per-agent failure isolation
                    error_details = {
                        "agent": agent_name,
                        "iteration": iteration,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "trace_id": context.trace_id,
                    }
                    logger.error(f"Agent {agent_name} failed", extra={"error_details": error_details})
                    
                    # Log structured failure
                    await self._log_execution(context, "agent_failure", error_details)
                    
                    # Continue orchestration despite agent failure
                    context.policy_violations.append({
                        "type": "agent_crash",
                        "agent": agent_name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Emit failure event (non-fatal)
                    await self._emit_event(streaming_callback, "agent_failure",
                                          agent=agent_name,
                                          error=str(e),
                                          recoverable=True)

        if not context.synthesized_answer:
            synthesizer = self.sub_agents[AgentType.SYNTHESIZER]
            try:
                context = await synthesizer.process(context, streaming_callback)
            except Exception as e:
                logger.error(f"Synthesizer failed: {e}", exc_info=True)
                context.policy_violations.append({
                    "type": "synthesis_crash",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        return context

    async def _stream_answer_chunks(self, context: SharedContext, streaming_callback):
        if streaming_callback is None:
            return
        answer = (context.synthesized_answer or "").strip()
        if not answer:
            return

        total_chunks = max(1, math.ceil(len(answer) / 80))
        await self._emit_event(streaming_callback, "answer_stream_start", total_chunks=total_chunks)
        for idx in range(0, len(answer), 80):
            chunk = answer[idx : idx + 80]
            await self._emit_event(
                streaming_callback,
                "answer_token_chunk",
                token=chunk,
                index=(idx // 80) + 1,
                total_chunks=total_chunks,
            )
        await self._emit_event(streaming_callback, "answer_stream_complete", total_chunks=total_chunks)

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
- DONE: No further work is needed

Return ONLY one of: DECOMPOSER, RAG, CRITIC, SYNTHESIZER, DONE.
"""

        response = await self.llm.generate(prompt, temperature=0.2, max_tokens=20)
        decision = response.get("text", "").strip().upper()

        agent_map = {
            "DECOMPOSER": AgentType.DECOMPOSER,
            "RAG": AgentType.RAG,
            "CRITIC": AgentType.CRITIC,
            "SYNTHESIZER": AgentType.SYNTHESIZER,
            "DONE": None
        }

        if decision == "DECOMPOSER" and context.dependency_graph is not None:
            logger.info("LLM attempted to re-run decomposer after decomposition. Routing to fallback agent.")
            if not context.retrieved_chunks:
                decision = "RAG"
            elif context.claims and not context.critiques:
                decision = "CRITIC"
            elif context.claims and not context.synthesized_answer:
                decision = "SYNTHESIZER"
            else:
                return None

        if decision in agent_map:
            if agent_map[decision] is None:
                return None
            return RoutingDecision(
                next_agent=agent_map[decision],
                reasoning=f"LLM decided: {decision}",
                priority=9,
                context_budget_allocation=2000,
                expected_output_type="varies"
            )

        # Fallback deterministic routing if LLM output is unclear
        if context.dependency_graph is None:
            fallback = AgentType.DECOMPOSER
            reasoning = "Fallback: no decomposition found"
        elif not context.retrieved_chunks:
            fallback = AgentType.RAG
            reasoning = "Fallback: no retrieved chunks yet"
        elif context.claims and not context.critiques:
            fallback = AgentType.CRITIC
            reasoning = "Fallback: claims exist without critique"
        elif context.claims and not context.synthesized_answer:
            fallback = AgentType.SYNTHESIZER
            reasoning = "Fallback: claims exist and final answer missing"
        else:
            return None

        logger.info("Orchestrator fallback selected %s", fallback.value)
        return RoutingDecision(
            next_agent=fallback,
            reasoning=reasoning,
            priority=1,
            context_budget_allocation=1500,
            expected_output_type="varies"
        )
    
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
            budget.remaining_tokens = max(0, tokens - budget.used_tokens)