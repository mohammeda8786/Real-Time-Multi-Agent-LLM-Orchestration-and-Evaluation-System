"""
Demo orchestration run. Uses platform-safe console setup before loading Groq/Chroma.
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.platform.runtime import (
    configure_runtime_warnings,
    configure_stdio_utf8,
    enforce_supported_python,
    log_startup_stage,
    stage_timer,
)

configure_stdio_utf8()
configure_runtime_warnings()
logging.basicConfig(level=logging.INFO)
enforce_supported_python()


async def main():
    t0 = stage_timer()
    log_startup_stage("demo_import_orchestrator_begin")

    from app.agents.orchestrator import OrchestratorAgent
    from app.models.schemas import SharedContext

    log_startup_stage(
        "demo_import_orchestrator_done",
        latency_ms=(stage_timer() - t0) * 1000,
    )

    orchestrator = OrchestratorAgent()
    query = (
        "Compare the effectiveness of reinforcement learning versus supervised learning "
        "for robotics applications"
    )
    context = SharedContext(original_query=query)

    async def stream_callback(event_type: str, data: dict):
        if "token" in data:
            print(data["token"], end="", flush=True)
        else:
            msg = data.get("message", data)
            print(f"\n[EVENT] {event_type}: {msg}")

    print("Running orchestration with streaming events...\n")
    t_run = stage_timer()
    result = await orchestrator.process(context, stream_callback)
    log_startup_stage(
        "demo_orchestration_complete",
        latency_ms=(stage_timer() - t_run) * 1000,
    )

    print("\n" + "=" * 50)
    print("FINAL ANSWER:")
    print("=" * 50)
    print(result.synthesized_answer or "No answer generated")

    if result.provenance_map:
        print("\n" + "=" * 50)
        print("PROVENANCE (sample):")
        print("=" * 50)
        for link in result.provenance_map[:3]:
            print(f"\nSentence: {(link.sentence or '')[:100]}...")
            print(f"Source Agent: {link.source_agent.value}")
            if link.source_chunks:
                print(f"Source Chunks: {', '.join(link.source_chunks[:2])}")

    print("\n" + "=" * 50)
    print(f"Status: {result.status}")
    print(f"Job / trace id: {result.job_id}")
    print(f"Claims: {len(result.claims)} | Critiques: {len(result.critiques)}")
    print(f"Policy violations: {len(result.policy_violations)} | Tool events: {len(result.tool_audit)}")


if __name__ == "__main__":
    asyncio.run(main())
