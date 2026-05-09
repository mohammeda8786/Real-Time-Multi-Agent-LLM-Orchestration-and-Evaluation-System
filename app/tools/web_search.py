"""
Web Search Tool - Returns structured results with sources and relevance scores
"""

from app.tools.base_tool import BaseTool, ToolResult
from typing import List, Dict
import asyncio
import hashlib
import json

class WebSearchTool(BaseTool):
    """Search tool that returns structured results with relevance scoring"""
    
    def __init__(self):
        super().__init__("web_search", max_retries=2)
        # Mock search results database
        self.mock_results = {
            "python": [
                {
                    "title": "Python Official Website",
                    "url": "https://www.python.org",
                    "snippet": "Python is a high-level programming language...",
                    "relevance": 0.95
                },
                {
                    "title": "Python Tutorial",
                    "url": "https://docs.python.org/3/tutorial/",
                    "snippet": "Learn Python from the official documentation...",
                    "relevance": 0.90
                },
                {
                    "title": "Stack Overflow - Python",
                    "url": "https://stackoverflow.com/questions/tagged/python",
                    "snippet": "Q&A community for Python developers...",
                    "relevance": 0.85
                }
            ],
            "machine learning": [
                {
                    "title": "Scikit-learn Documentation",
                    "url": "https://scikit-learn.org",
                    "snippet": "Machine learning library for Python...",
                    "relevance": 0.92
                },
                {
                    "title": "TensorFlow Guide",
                    "url": "https://www.tensorflow.org",
                    "snippet": "End-to-end open source platform for ML...",
                    "relevance": 0.90
                }
            ]
        }
    
    async def execute(self, query: str = None, limit: int = 5, **kwargs) -> ToolResult:
        """Execute web search"""
        if not query or not isinstance(query, str):
            return self._handle_malformed_input("query must be a non-empty string")
        
        query_lower = query.lower()
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Simulate network latency
            await asyncio.sleep(0.1)
            
            # Find matching results
            results = []
            for key in self.mock_results:
                if key in query_lower:
                    results.extend(self.mock_results[key])
            
            if not results:
                return self._handle_empty_results()
            
            # Sort by relevance and limit
            results = sorted(results, key=lambda x: x['relevance'], reverse=True)[:limit]
            
            end_time = asyncio.get_event_loop().time()
            latency_ms = (end_time - start_time) * 1000
            
            output_json = json.dumps(results)
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": results,
                    "count": len(results),
                    "search_time_ms": latency_ms
                },
                latency_ms=latency_ms,
                input_hash=hashlib.md5(query.encode()).hexdigest(),
                output_hash=hashlib.md5(output_json.encode()).hexdigest()
            )
        except asyncio.TimeoutError:
            return self._handle_timeout(5)
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e)
            )
