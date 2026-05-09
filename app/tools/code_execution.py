"""
Code Execution Tool - Runs Python snippets with timeout and error handling
"""

from app.tools.base_tool import BaseTool, ToolResult
import asyncio
import subprocess
import hashlib
import json
import tempfile
import os

class CodeExecutionTool(BaseTool):
    """Execute Python code in a sandbox with timeout"""
    
    def __init__(self, timeout: int = 5):
        super().__init__("code_execution", max_retries=1)
        self.timeout = timeout
    
    async def execute(self, code: str = None, **kwargs) -> ToolResult:
        """Execute Python code and return result"""
        if not code or not isinstance(code, str):
            return self._handle_malformed_input("code must be a non-empty string")
        
        # Safety check: reject dangerous operations
        dangerous_ops = ["os.system", "exec(", "eval(", "__import__", "open("]
        if any(op in code for op in dangerous_ops):
            return ToolResult(
                success=False,
                data=None,
                error="Code contains forbidden operations for security"
            )
        
        try:
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Run with timeout
                result = await asyncio.wait_for(
                    self._run_code(temp_file),
                    timeout=self.timeout
                )
                return result
            finally:
                # Cleanup
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        except asyncio.TimeoutError:
            return self._handle_timeout(self.timeout)
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Execution failed: {str(e)}"
            )
    
    async def _run_code(self, temp_file: str) -> ToolResult:
        """Run code file and capture output"""
        loop = asyncio.get_event_loop()
        
        def run():
            try:
                result = subprocess.run(
                    ["python", temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                return result
            except subprocess.TimeoutExpired:
                return None
        
        result = await loop.run_in_executor(None, run)
        
        if result is None:
            return self._handle_timeout(self.timeout)
        
        output = result.stdout + result.stderr
        return ToolResult(
            success=result.returncode == 0,
            data={
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "output": output
            },
            error=result.stderr if result.returncode != 0 else None,
            output_hash=hashlib.md5(output.encode()).hexdigest()
        )
