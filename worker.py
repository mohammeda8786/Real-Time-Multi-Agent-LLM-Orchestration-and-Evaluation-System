"""
Background Worker - Processes evaluation jobs and prompt rewrites
"""

import asyncio
import json
import logging
from datetime import datetime

from app.platform.runtime import (
    configure_runtime_warnings,
    configure_stdio_utf8,
    warn_unsupported_python,
)

configure_stdio_utf8()
configure_runtime_warnings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warn_unsupported_python()

from app.agents.orchestrator import OrchestratorAgent
from app.evaluation.pipeline import EvaluationPipeline
from app.meta.prompt_optimizer import SelfImprovingPromptLoop

class BackgroundWorker:
    """Process long-running tasks asynchronously"""
    
    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.eval_pipeline = EvaluationPipeline()
        self.prompt_optimizer = SelfImprovingPromptLoop()
    
    async def run_evaluation(self):
        """Run full evaluation pipeline on all 15 test cases"""
        logger.info("Starting evaluation run...")
        
        try:
            results = await self.eval_pipeline.run_evaluation(self.orchestrator)
            
            logger.info(f"Evaluation complete: {json.dumps(results, indent=2)}")
            
            # Analyze failures and propose rewrites
            proposals = await self.prompt_optimizer.analyze_failures(self.eval_pipeline)
            
            logger.info(f"Generated {len(proposals)} prompt rewrite proposals")
            for prop in proposals:
                logger.info(f"  - {prop.proposal_id}: {prop.target_dimension}")
            
            return results
        
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return None
    
    async def retrigger_failed_tests(self):
        """Re-evaluate only failed test cases"""
        logger.info("Retriggering evaluation on failed tests...")
        
        try:
            failing_tests = self.eval_pipeline.get_failing_tests()
            logger.info(f"Found {len(failing_tests)} failing tests to retry")
            
            # In production: filter pipeline to only run failing tests
            # For now, run full pipeline again
            results = await self.eval_pipeline.run_evaluation(self.orchestrator)
            
            return results
        
        except Exception as e:
            logger.error(f"Retrigger failed: {e}", exc_info=True)
            return None
    
    async def apply_approved_rewrites(self):
        """Apply all approved prompt rewrites and re-evaluate"""
        logger.info("Applying approved prompt rewrites...")
        
        try:
            deltas = await self.prompt_optimizer.apply_approved_rewrites()
            
            logger.info(f"Applied {len(deltas)} rewrites")
            for proposal_id, delta in deltas.items():
                logger.info(f"  - {proposal_id}: +{delta:.2%} improvement")
            
            # Re-run evaluation with new prompts
            results = await self.eval_pipeline.run_evaluation(self.orchestrator)
            
            return results
        
        except Exception as e:
            logger.error(f"Rewrite application failed: {e}", exc_info=True)
            return None

async def main():
    """Worker event loop"""
    worker = BackgroundWorker()
    
    logger.info("="*50)
    logger.info("BACKGROUND WORKER STARTED")
    logger.info("="*50)
    
    # Periodic tasks
    eval_interval = 3600  # Run evaluation every hour
    
    while True:
        try:
            # Check for pending tasks (from API requests)
            # For now, run evaluation on schedule
            
            logger.info(f"[{datetime.now().isoformat()}] Waiting for tasks...")
            
            # Simulate task reception
            await asyncio.sleep(eval_interval)
            
            # Run evaluation
            await worker.run_evaluation()
        
        except KeyboardInterrupt:
            logger.info("Worker shutting down...")
            break
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            await asyncio.sleep(30)  # Wait before retry

if __name__ == "__main__":
    asyncio.run(main())
