from typing import Dict, Any, Optional
from app.models.schemas import SharedContext
import asyncio
import json

class SharedContextManager:
    """
    Manages shared context object.
    All agents read/write to this - no direct agent-to-agent calls.
    Orchestrator mediates all handoffs.
    """
    
    def __init__(self):
        # In production: use Redis for distributed storage
        self._contexts: Dict[str, SharedContext] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
    
    async def get_context(self, job_id: str) -> Optional[SharedContext]:
        """Retrieve context by job ID"""
        async with self._get_lock(job_id):
            return self._contexts.get(job_id)
    
    async def update_context(self, context: SharedContext) -> SharedContext:
        """Update entire context"""
        async with self._get_lock(context.job_id):
            self._contexts[context.job_id] = context
            return context
    
    async def patch_context(self, job_id: str, updates: Dict[str, Any]) -> SharedContext:
        """Partial update to context"""
        async with self._get_lock(job_id):
            context = self._contexts.get(job_id)
            if context:
                for key, value in updates.items():
                    if hasattr(context, key):
                        setattr(context, key, value)
                self._contexts[job_id] = context
                return context
        raise ValueError(f"Context {job_id} not found")
    
    async def _get_lock(self, job_id: str) -> asyncio.Lock:
        """Get or create lock for job_id"""
        if job_id not in self._locks:
            self._locks[job_id] = asyncio.Lock()
        return self._locks[job_id]
    
    async def delete_context(self, job_id: str):
        """Clean up context after completion"""
        async with self._get_lock(job_id):
            if job_id in self._contexts:
                del self._contexts[job_id]
            if job_id in self._locks:
                del self._locks[job_id]