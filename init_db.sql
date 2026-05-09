-- PostgreSQL bootstrap for Mega.AI (optional; app defaults to SQLite for jobs/eval).
CREATE TABLE IF NOT EXISTS orchestration_events (
    id BIGSERIAL PRIMARY KEY,
    job_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_orchestration_job ON orchestration_events (job_id);
CREATE INDEX IF NOT EXISTS idx_orchestration_trace ON orchestration_events (trace_id);
