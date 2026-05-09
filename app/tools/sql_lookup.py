"""
SQL Lookup Tool - Convert natural language to SQL and query local database
"""

from app.tools.base_tool import BaseTool, ToolResult
from typing import List, Dict
import asyncio
import hashlib
import json
import sqlite3

class SQLLookupTool(BaseTool):
    """Query structured data via natural language to SQL conversion"""
    
    def __init__(self, db_path: str = "local_knowledge.db"):
        super().__init__("sql_lookup", max_retries=2)
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize mock database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if not exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                entity_type TEXT,
                description TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY,
                entity1_id INTEGER,
                entity2_id INTEGER,
                relation_type TEXT,
                details TEXT
            )
        ''')
        
        # Insert sample data
        cursor.execute("DELETE FROM entities")
        cursor.execute("DELETE FROM relations")
        
        sample_entities = [
            ("Python", "language", "High-level programming language"),
            ("Java", "language", "Object-oriented programming language"),
            ("TensorFlow", "library", "Machine learning platform"),
            ("PyTorch", "library", "ML framework"),
        ]
        
        for name, entity_type, desc in sample_entities:
            try:
                cursor.execute(
                    "INSERT INTO entities (name, entity_type, description) VALUES (?, ?, ?)",
                    (name, entity_type, desc)
                )
            except sqlite3.IntegrityError:
                pass
        
        conn.commit()
        conn.close()
    
    async def execute(self, nl_query: str = None, **kwargs) -> ToolResult:
        """Execute natural language query converted to SQL"""
        if not nl_query or not isinstance(nl_query, str):
            return self._handle_malformed_input("nl_query must be non-empty string")
        
        try:
            # Convert NL to SQL (simplified)
            sql = self._nl_to_sql(nl_query)
            if not self._is_safe_select(sql):
                return ToolResult(
                    success=False,
                    data=None,
                    error="Only single-statement SELECT queries are allowed",
                )

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return self._handle_empty_results()
            
            output = json.dumps([list(row) for row in results])
            return ToolResult(
                success=True,
                data={
                    "query": nl_query,
                    "sql": sql,
                    "results": [list(row) for row in results],
                    "count": len(results)
                },
                input_hash=hashlib.md5(nl_query.encode()).hexdigest(),
                output_hash=hashlib.md5(output.encode()).hexdigest()
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Query failed: {str(e)}"
            )
    
    def _nl_to_sql(self, nl_query: str) -> str:
        """Convert natural language to SQL (simplified)"""
        nl_lower = nl_query.lower()
        
        if "language" in nl_lower and "type" in nl_lower:
            return "SELECT name, entity_type, description FROM entities WHERE entity_type='language'"
        elif "library" in nl_lower:
            return "SELECT name, entity_type, description FROM entities WHERE entity_type='library'"
        else:
            return "SELECT * FROM entities LIMIT 10"

    def _is_safe_select(self, sql: str) -> bool:
        s = (sql or "").strip().lower()
        if not s.startswith("select"):
            return False
        banned = (";", "attach", "pragma", "delete", "insert", "update", "drop", "create", "replace")
        return not any(b in s for b in banned)
