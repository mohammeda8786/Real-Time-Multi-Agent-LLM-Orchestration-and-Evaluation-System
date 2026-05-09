"""
Persistence Layer - SQLite Storage
"""

import sqlite3
import json
from datetime import datetime
import os

class PersistenceStore:
    def __init__(self, db_path: str = "agent_system.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Jobs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                query TEXT,
                result TEXT,
                status TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        # Evaluations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                run_id TEXT PRIMARY KEY,
                results TEXT,
                summary TEXT,
                timestamp TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompt_rewrites (
                proposal_id TEXT PRIMARY KEY,
                payload TEXT,
                updated_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_job_result(self, job_id: str, query: str, result: dict, status: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO jobs (job_id, query, result, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM jobs WHERE job_id=?), ?), ?)
        ''', (job_id, query, json.dumps(result), status, job_id, now, now))
        
        conn.commit()
        conn.close()
    
    def get_job(self, job_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT query, result, status, updated_at FROM jobs WHERE job_id = ?', (job_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "query": row[0],
                "result": json.loads(row[1]),
                "status": row[2],
                "updated_at": row[3]
            }
        return None
    
    def get_latest_evaluation(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT summary, timestamp FROM evaluations ORDER BY timestamp DESC LIMIT 1')
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        return None

    def save_evaluation(self, run_id: str, summary: dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        payload = json.dumps(summary, default=str)
        cursor.execute(
            '''
            INSERT OR REPLACE INTO evaluations (run_id, results, summary, timestamp)
            VALUES (?, ?, ?, ?)
            ''',
            (run_id, payload, payload, now),
        )
        conn.commit()
        conn.close()

    def save_prompt_rewrite(self, proposal: dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        pid = proposal.get("proposal_id", "unknown")
        cursor.execute(
            '''
            INSERT OR REPLACE INTO prompt_rewrites (proposal_id, payload, updated_at)
            VALUES (?, ?, ?)
            ''',
            (pid, json.dumps(proposal, default=str), now),
        )
        conn.commit()
        conn.close()

    def get_prompt_rewrite(self, proposal_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT payload FROM prompt_rewrites WHERE proposal_id = ?", (proposal_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
        return None