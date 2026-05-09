from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.sql import func
from app.db.session import Base

class Job(Base):
    __tablename__ = "jobs"
    id = Column(String, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="initialized")
    answer = Column(Text, nullable=True)
    stats = Column(SQLiteJSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    trace_events = relationship("TraceEvent", back_populates="job")

class TraceEvent(Base):
    __tablename__ = "trace_events"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    agent = Column(String, nullable=False)
    event_type = Column(String, nullable=False)
    details = Column(SQLiteJSON, nullable=True)
    job = relationship("Job", back_populates="trace_events")

class EvalRun(Base):
    __tablename__ = "eval_runs"
    id = Column(String, primary_key=True, index=True)
    summary = Column(SQLiteJSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class PromptRewrite(Base):
    __tablename__ = "prompt_rewrites"
    id = Column(String, primary_key=True, index=True)
    target_agent = Column(String, nullable=False)
    target_dimension = Column(String, nullable=False)
    original_prompt = Column(Text, nullable=True)
    proposed_prompt = Column(Text, nullable=True)
    justification = Column(Text, nullable=True)
    expected_improvement = Column(Float, nullable=True)
    status = Column(String, nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    approved_at = Column(DateTime(timezone=True), nullable=True)
    performance_delta = Column(Float, nullable=True)
    metadata = Column(SQLiteJSON, nullable=True)
