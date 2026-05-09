from typing import List, Dict, Optional, Any
"""
Decomposer - Uses LLM to break queries into tasks with dependencies
"""

from app.agents.base import BaseAgent
from app.models.schemas import SharedContext, DependencyGraph, SubTask, SubTaskStatus, AgentType
from app.llm_client import LLMClient
import json
import uuid

class DecompositionAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.DECOMPOSER)
        self.llm = LLMClient()
    
    async def process(self, context: SharedContext, streaming_callback=None):
        await self._emit_event(streaming_callback, "decomposer_start",
                              message="LLM is breaking down the query into tasks")
        
        # Use LLM to identify sub-tasks with dependencies
        sub_tasks = await self._llm_identify_sub_tasks(context.original_query)
        
        # Build dependency graph
        dependency_graph = DependencyGraph(
            original_query=context.original_query,
            root_task_id=sub_tasks[0].task_id if sub_tasks else ""
        )
        
        for task in sub_tasks:
            dependency_graph.add_task(task)
        
        context.dependency_graph = dependency_graph
        
        await self._emit_event(streaming_callback, "decomposer_complete",
                              task_count=len(sub_tasks))
        
        return context
    
    async def _llm_identify_sub_tasks(self, query: str):
        """Use LLM to identify sub-tasks and their dependencies"""
        
        prompt = f"""Break this query into sub-tasks with dependencies:

Query: "{query}"

Return JSON exactly like this:
{{
    "tasks": [
        {{"id": 1, "description": "First task", "task_type": "retrieval", "depends_on": []}},
        {{"id": 2, "description": "Second task", "task_type": "reasoning", "depends_on": [1]}}
    ]
}}

Task types: retrieval, reasoning, verification, synthesis
"""
        
        response = await self.llm.generate(prompt, temperature=0.3, max_tokens=500)
        
        try:
            # Extract JSON from response
            data = json.loads(response["text"])
            tasks = []
            
            for task_data in data.get("tasks", []):
                task = SubTask(
                    task_id=str(uuid.uuid4()),
                    description=task_data["description"],
                    task_type=task_data["task_type"],
                    dependencies=[f"task_{d}" for d in task_data.get("depends_on", [])]
                )
                tasks.append(task)
            
            return tasks
        except:
            # Fallback to default decomposition
            return self._default_decomposition(query)
    
    def _default_decomposition(self, query: str):
        """Fallback when LLM fails"""
        task1 = SubTask(
            task_id=str(uuid.uuid4()),
            description=f"Retrieve information about: {query}",
            task_type="retrieval",
            dependencies=[]
        )
        return [task1]