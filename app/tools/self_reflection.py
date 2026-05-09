"""
Self-Reflection Tool - Agent can review its own outputs and identify contradictions
"""

from app.tools.base_tool import BaseTool, ToolResult
from typing import Optional, Dict, Any
import hashlib
import json

class SelfReflectionTool(BaseTool):
    """Allows agents to self-review outputs and detect contradictions"""
    
    def __init__(self):
        super().__init__("self_reflection", max_retries=1)
        self.session_history: Dict[str, list] = {}
    
    async def execute(self, agent_id: str = None, session_id: str = None, **kwargs) -> ToolResult:
        """Retrieve agent's own execution history within session"""
        if not agent_id or not session_id:
            return self._handle_malformed_input("agent_id and session_id required")
        
        key = f"{session_id}:{agent_id}"
        
        if key not in self.session_history:
            return self._handle_empty_results()
        
        history = self.session_history[key]
        
        # Analyze for contradictions
        contradictions = self._detect_contradictions(history)
        
        output = json.dumps(history)
        return ToolResult(
            success=True,
            data={
                "agent_id": agent_id,
                "session_id": session_id,
                "execution_count": len(history),
                "history": history,
                "contradictions": contradictions,
                "has_contradictions": len(contradictions) > 0
            },
            output_hash=hashlib.md5(output.encode()).hexdigest()
        )
    
    def record_execution(self, agent_id: str, session_id: str, output: Any, metadata: Optional[Dict] = None):
        """Record agent execution for later reflection"""
        key = f"{session_id}:{agent_id}"
        
        if key not in self.session_history:
            self.session_history[key] = []
        
        self.session_history[key].append({
            "output": output,
            "metadata": metadata or {},
            "timestamp": __import__('datetime').datetime.now().isoformat()
        })
    
    def _detect_contradictions(self, history: list) -> list:
        """Detect logical contradictions in history"""
        contradictions = []
        
        if len(history) < 2:
            return contradictions
        
        # Simple contradiction detection: look for claim reversals
        for i in range(len(history) - 1):
            prev_output = str(history[i].get("output", "")).lower()
            curr_output = str(history[i + 1].get("output", "")).lower()
            
            # Detect yes/no reversals
            if ("yes" in prev_output and "no" in curr_output) or \
               ("no" in prev_output and "yes" in curr_output):
                contradictions.append({
                    "step": i,
                    "type": "boolean_reversal",
                    "severity": "high"
                })
            
            # Detect statement reversals
            if prev_output and curr_output:
                if any(negation in curr_output for negation in ["not ", "never ", "impossible"]) \
                   and prev_output.replace("not ", "").replace("never ", "") in curr_output:
                    contradictions.append({
                        "step": i,
                        "type": "negation_reversal",
                        "severity": "medium"
                    })
        
        return contradictions
    
    def clear_session(self, session_id: str):
        """Clear history for a session"""
        keys_to_delete = [k for k in self.session_history if k.startswith(f"{session_id}:")]
        for key in keys_to_delete:
            del self.session_history[key]
