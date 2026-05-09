from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum
import uuid

class AgentType(str, Enum):
    ORCHESTRATOR = "orchestrator"
    DECOMPOSER = "decomposer"
    RAG = "rag"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"

class SubTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

class SubTask(BaseModel):
    task_id: str
    description: str
    task_type: Literal["retrieval", "reasoning", "calculation", "verification", "synthesis"]
    dependencies: List[str] = []  # task_ids that must complete first
    status: SubTaskStatus = SubTaskStatus.PENDING
    result: Optional[Any] = None
    assigned_agent: Optional[AgentType] = None
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

class DependencyGraph(BaseModel):
    original_query: str
    root_task_id: str
    tasks: Dict[str, SubTask] = {}
    execution_order: List[str] = []
    
    def add_task(self, task: SubTask):
        self.tasks[task.task_id] = task
    
    def get_ready_tasks(self) -> List[SubTask]:
        """Returns tasks whose dependencies are all completed"""
        ready = []
        for task in self.tasks.values():
            if task.status == SubTaskStatus.PENDING:
                deps_completed = all(
                    self.tasks[dep_id].status == SubTaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                if deps_completed:
                    ready.append(task)
        return ready
    
    def mark_completed(self, task_id: str, result: Any):
        if task_id in self.tasks:
            self.tasks[task_id].status = SubTaskStatus.COMPLETED
            self.tasks[task_id].result = result
            self.tasks[task_id].completed_at = datetime.now()

class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    source: str
    relevance_score: float
    retrieval_step: int  # 1 for first hop, 2 for second hop, etc.

class Claim(BaseModel):
    claim_id: str
    text: str
    agent_source: AgentType
    chunk_citations: List[str] = []  # chunk_ids that support this claim
    confidence_score: float = Field(ge=0, le=1)
    start_char: Optional[int] = None
    end_char: Optional[int] = None

class Critique(BaseModel):
    claim_id: str
    disagree_text: str  # Specific span the critic disagrees with
    disagreement_reason: str
    suggested_correction: Optional[str] = None
    confidence_in_critique: float = Field(ge=0, le=1)

class ProvenanceLink(BaseModel):
    sentence: str
    source_agent: AgentType
    source_chunks: List[str] = []
    supporting_claims: List[str] = []

class ContextBudget(BaseModel):
    agent_type: AgentType
    allocated_tokens: int
    used_tokens: int = 0
    remaining_tokens: int

class RoutingDecision(BaseModel):
    next_agent: AgentType
    reasoning: str
    priority: int = 1
    context_budget_allocation: int
    expected_output_type: str

class SharedContext(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str
    status: str = "initialized"
    
    # Routing history
    routing_decisions: List[RoutingDecision] = []
    execution_trace: List[Dict[str, Any]] = []
    
    # Decomposition
    dependency_graph: Optional[DependencyGraph] = None
    
    # Retrieval (multi-hop)
    retrieved_chunks: List[RetrievedChunk] = []
    multi_hop_reasoning_path: List[Dict[str, Any]] = []
    
    # Claims and critiques
    claims: List[Claim] = []
    critiques: List[Critique] = []
    
    # Final output
    synthesized_answer: Optional[str] = None
    provenance_map: List[ProvenanceLink] = []
    
    # Budget tracking
    budgets: Dict[AgentType, ContextBudget] = {}
    
    # Policy violations
    policy_violations: List[Dict[str, Any]] = []
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }