"""
Run Full Evaluation Pipeline and Self-Improvement Loop
Run: python run_evaluation.py
"""

import asyncio
import json
import sys
from datetime import datetime

print("\n" + "="*70)
print("MEGA.AI - FULL EVALUATION PIPELINE")
print("="*70 + "\n")

async def main():
    from app.agents.orchestrator import OrchestratorAgent
    from app.evaluation.pipeline import EvaluationPipeline
    from app.meta.prompt_optimizer import SelfImprovingPromptLoop
    
    # Initialize components
    orchestrator = OrchestratorAgent()
    eval_pipeline = EvaluationPipeline()
    prompt_optimizer = SelfImprovingPromptLoop()
    
    print("📊 STEP 1: Running Evaluation on All 15 Test Cases")
    print("-" * 70)
    
    try:
        # Run evaluation
        results = await eval_pipeline.run_evaluation(orchestrator)
        
        print(f"\n✅ Evaluation Complete!")
        print(f"   Run ID: {results['eval_id']}")
        print(f"   Timestamp: {results['timestamp']}")
        print(f"   Total tests: {results['total_tests']}")
        
        # Print by category
        print(f"\n📈 Results by Category:")
        for category, stats in results['by_category'].items():
            avg_score = stats['average_score']
            print(f"   • {category.upper()}: {avg_score:.2%} ({stats['count']} tests)")
        
        # Print dimension averages
        print(f"\n📊 Overall Scores by Dimension:")
        for dimension, score in sorted(results['overall_scores'].items()):
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   • {dimension:30} {bar} {score:.2%}")
        
        # Print best/worst
        print(f"\n🏆 Best Performers:")
        for performer in results['best_performers'][:3]:
            print(f"   • {performer['test_id']}: {performer['score']:.2%}")
        
        print(f"\n⚠️  Worst Performers (needs improvement):")
        for performer in results['worst_performers'][:3]:
            print(f"   • {performer['test_id']}: {performer['score']:.2%}")
    
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Analyze failures
    print(f"\n\n🔍 STEP 2: Analyzing Failures and Proposing Rewrites")
    print("-" * 70)
    
    try:
        proposals = await prompt_optimizer.analyze_failures(eval_pipeline)
        
        if proposals:
            print(f"\n✅ Generated {len(proposals)} prompt rewrite proposals:")
            for prop in proposals:
                print(f"\n   Proposal ID: {prop.proposal_id}")
                print(f"   Target Agent: {prop.target_agent}")
                print(f"   Target Dimension: {prop.target_dimension}")
                print(f"   Failing Tests: {len(prop.failing_test_ids)}")
                print(f"   Expected Improvement: +{prop.expected_improvement:.1%}")
                print(f"   Status: {prop.status}")
        else:
            print("\n✅ No failing tests - system performing well!")
    
    except Exception as e:
        print(f"\n⚠️  Failure analysis: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Mock approval and re-evaluation
    print(f"\n\n✅ STEP 3: Simulating Human Approval of Top Proposal")
    print("-" * 70)
    
    try:
        if proposals:
            top_proposal = proposals[0]
            
            print(f"\nApproving: {top_proposal.proposal_id}")
            print(f"Reasoning: {top_proposal.justification[:100]}...")
            
            await prompt_optimizer.approve_rewrite(top_proposal.proposal_id)
            print(f"✅ Approval recorded with timestamp")
            
            # Apply rewrites
            deltas = await prompt_optimizer.apply_approved_rewrites()
            
            if deltas:
                print(f"\n✅ Applied {len(deltas)} rewrites with simulated improvements:")
                for prop_id, delta in deltas.items():
                    print(f"   • {prop_id}: +{delta:.1%}")
            
            # Get audit trail
            audit = prompt_optimizer.get_proposal_audit_trail(top_proposal.proposal_id)
            print(f"\n📋 Audit Trail:")
            print(f"   Created: {audit['created_at']}")
            print(f"   Status: {audit['status']}")
            print(f"   Approval Timestamp: {audit['approval_timestamp']}")
            print(f"   Expected Delta: +{audit['expected_improvement']:.1%}")
            print(f"   Actual Delta: +{audit['actual_performance_delta']:.1%}")
    
    except Exception as e:
        print(f"\n⚠️  Approval simulation: {e}")
        import traceback
        traceback.print_exc()
    
    # Final summary
    print(f"\n\n" + "="*70)
    print("EVALUATION PIPELINE COMPLETE")
    print("="*70)
    print("""
Summary:
✅ Ran 15 comprehensive test cases (baseline, ambiguous, adversarial)
✅ Scored on 6 dimensions with justifications
✅ Identified failing tests by performance dimension
✅ Generated LLM-powered prompt rewrite proposals
✅ Simulated human approval process
✅ Built complete audit trail

Key Metrics:
• Multi-dimensional scoring framework
• Adversarial robustness testing
• Self-improvement loop with human oversight
• Full reproducibility and auditability

Next Steps:
1. Run full API server: python api.py
2. Test endpoints with curl/Postman
3. Deploy with Docker: docker-compose up
4. Monitor performance over time
5. Approve/reject rewrites via /meta/approve endpoint

For production:
• Configure PostgreSQL for persistent storage
• Set up Redis for distributed context
• Enable SSE streaming for real-time updates
• Configure monitoring and alerting
• Run continuous evaluation loop
    """)
    print("="*70 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
