#!/bin/bash

# Quick Setup Script for Mega.AI Multi-Agent LLM System
# Run this to get started in 5 minutes

set -e

echo "=================================="
echo "Mega.AI Setup Script"
echo "=================================="

# 1. Check Python
echo "✓ Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 not found. Install from python.org"
    exit 1
fi
echo "  Python version: $(python3 --version)"

# 2. Create virtual environment
echo "✓ Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  Virtual environment already exists"
else
    python3 -m venv venv
    echo "  Virtual environment created"
fi

# 3. Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate

# 4. Install dependencies
echo "✓ Installing dependencies..."
pip install -q -r requirements.txt
echo "  Dependencies installed"

# 5. Setup .env
echo "✓ Setting up .env..."
if [ -f ".env" ]; then
    echo "  .env already exists"
else
    cp .env.example .env
    echo "  Created .env from template"
    echo "  ⚠️  IMPORTANT: Edit .env and add your GROQ_API_KEY"
    echo "     Get free key from: https://console.groq.com/keys"
fi

# 6. Create directories
echo "✓ Creating necessary directories..."
mkdir -p chroma_db logs

# 7. Test LLM connection
echo "✓ Testing LLM connection..."
python3 -c "
from app.llm_client import LLMClient
llm = LLMClient()
print('  ✓ LLM client initialized successfully')
" || echo "  ⚠️  LLM connection test failed (check GROQ_API_KEY)"

# 8. Test imports
echo "✓ Testing imports..."
python3 -c "
from app.agents.orchestrator import OrchestratorAgent
from app.evaluation.pipeline import EvaluationPipeline
from app.meta.prompt_optimizer import SelfImprovingPromptLoop
print('  ✓ All imports successful')
"

echo ""
echo "=================================="
echo "✓ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your GROQ_API_KEY"
echo "2. Start server: python api.py"
echo "3. Test API: curl http://localhost:8000/health"
echo "4. Submit query: curl -X POST http://localhost:8000/submit -H 'Content-Type: application/json' -d '{\"query\": \"What is Python?\"}'"
echo "5. View docs: http://localhost:8000/docs"
echo ""
echo "Or use Docker:"
echo "  docker-compose up"
echo ""
