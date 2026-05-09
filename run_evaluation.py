"""
Run Full Evaluation Pipeline and Self-Improvement Loop
Run: python run_evaluation.py
"""

import asyncio
import logging
import sys
from datetime import datetime

from app.platform.runtime import configure_runtime_warnings, configure_stdio_utf8, warn_unsupported_python

configure_stdio_utf8()
configure_runtime_warnings()
logging.basicConfig(level=logging.INFO)
warn_unsupported_python()

print("\n" + "=" * 70)
print("MEGA.AI - FULL EVALUATION PIPELINE")
print("=" * 70 + "\n")


async def main():
    from app.agents.orchestrator import OrchestratorAgent
    from app.evaluation.pipeline import EvaluationPipeline
    from app.meta.prompt_optimizer import SelfImprovingPromptLoop

    orchestrator = OrchestratorAgent()
    eval_pipeline = EvaluationPipeline()
    prompt_optimizer = SelfImprovingPromptLoop()
    proposals = []

    print("[STEP 1] Running evaluation on all 15 test cases")
    print("-" * 70)

    try:
        results = await eval_pipeline.run_evaluation(orchestrator)

        print("\n[OK] Evaluation complete")
        print(f"   Run ID: {results['eval_id']}")
        print(f"   Timestamp: {results['timestamp']}")
        print(f"   Total tests: {results['total_tests']}")

        print("\n[CATEGORY] Results by category:")
        for category, stats in results["by_category"].items():
            avg_score = stats["average_score"]
            print(f"   - {category.upper()}: {avg_score:.2%} ({stats['count']} tests)")

        print("\n[DIMENSIONS] Overall scores:")
        for dimension, score in sorted(results["overall_scores"].items()):
            bar_length = int(score * 20)
            bar = "#" * bar_length + "-" * (20 - bar_length)
            print(f"   - {dimension:30} {bar} {score:.2%}")

        print("\n[BEST] Top performers:")
        for performer in results["best_performers"][:3]:
            print(f"   - {performer['test_id']}: {performer['score']:.2%}")

        print("\n[WARN] Lowest performers:")
        for performer in results["worst_performers"][:3]:
            print(f"   - {performer['test_id']}: {performer['score']:.2%}")

        if results.get("failure_detected"):
            print(
                f"\n[WARN] Threshold failures: {results.get('failed_tests', 0)} test(s); see persisted eval cases[]."
            )
        else:
            print("\n[OK] All tests within configured thresholds.")

    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print(f"\n\n[STEP 2] Analyzing failures and proposing rewrites")
    print("-" * 70)

    try:
        proposals = await prompt_optimizer.analyze_failures(eval_pipeline)

        failing_tests = eval_pipeline.get_failing_tests()

        if proposals:
            print(f"\n[OK] Generated {len(proposals)} prompt rewrite proposals:")
            for prop in proposals:
                print(f"\n   Proposal ID: {prop.proposal_id}")
                print(f"   Target Agent: {prop.target_agent}")
                print(f"   Target Dimension: {prop.target_dimension}")
                print(f"   Failing Tests: {len(prop.failing_test_ids)}")
                print(f"   Expected Improvement: +{prop.expected_improvement:.1%}")
                print(f"   Status: {prop.status}")
        elif failing_tests:
            print(f"\n[WARN] {len(failing_tests)} tests failed but no proposals generated")
            print("   Possible issues:")
            print("   - LLM client unavailable for proposal generation")
            print("   - All failures in unsupported dimensions")
            print("   - Proposal generation logic error")
        else:
            print("\n[OK] No failing tests under current thresholds.")

    except Exception as e:
        print(f"\n[WARN] Failure analysis: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n\n[STEP 3] Simulating human approval of top proposal")
    print("-" * 70)

    try:
        if proposals:
            top_proposal = proposals[0]

            print(f"\nApproving: {top_proposal.proposal_id}")
            print(f"Reasoning: {top_proposal.justification[:100]}...")

            await prompt_optimizer.approve_rewrite(top_proposal.proposal_id)
            print("[OK] Approval recorded with timestamp")

            deltas = await prompt_optimizer.apply_approved_rewrites()

            if deltas:
                print(f"\n[OK] Applied {len(deltas)} rewrites with simulated improvements:")
                for prop_id, delta in deltas.items():
                    print(f"   - {prop_id}: +{delta:.1%}")

            audit = prompt_optimizer.get_proposal_audit_trail(top_proposal.proposal_id)
            print(f"\n[AUDIT] Trail:")
            print(f"   Created: {audit['created_at']}")
            print(f"   Status: {audit['status']}")
            print(f"   Approval Timestamp: {audit['approval_timestamp']}")
            print(f"   Expected Delta: +{audit['expected_improvement']:.1%}")
            print(f"   Actual Delta: +{audit['actual_performance_delta']:.1%}")

    except Exception as e:
        print(f"\n[WARN] Approval simulation: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n\n" + "=" * 70)
    print("EVALUATION PIPELINE COMPLETE")
    print("=" * 70)
    print(
        """
Summary:
- Ran 15 test cases (baseline, ambiguous, adversarial)
- Scored on 6 dimensions with justifications
- Identified failing tests by performance dimension
- Generated LLM-powered prompt rewrite proposals where applicable
- Simulated human approval process
- Built audit trail (SQLite)

Next Steps:
1. Run API server: python api.py
2. Test endpoints with curl or /docs
3. Deploy: docker compose up --build
4. GET /diagnostics for runtime versions
5. Approve/reject rewrites via POST /meta/approve

Notes:
- Use Python 3.11 or 3.12 for best compatibility (see README).
"""
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
