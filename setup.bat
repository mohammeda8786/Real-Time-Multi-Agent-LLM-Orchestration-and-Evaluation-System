@echo off
REM Quick Setup Script for Mega.AI (Windows)
REM Run this to get started in 5 minutes

setlocal enabledelayedexpansion

echo.
echo ==================================
echo Mega.AI Setup Script (Windows)
echo ==================================
echo.

REM 1. Check Python
echo [1/8] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from python.org
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do set PYVER=%%i
echo OK - Python %PYVER%
echo.

REM 2. Create virtual environment
echo [2/8] Creating virtual environment...
if exist "venv" (
    echo OK - Virtual environment already exists
) else (
    python -m venv venv
    echo OK - Virtual environment created
)
echo.

REM 3. Activate virtual environment
echo [3/8] Activating virtual environment...
call venv\Scripts\activate.bat
echo OK - Virtual environment activated
echo.

REM 4. Install dependencies
echo [4/8] Installing dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)
echo OK - Dependencies installed
echo.

REM 5. Setup .env
echo [5/8] Setting up .env...
if exist ".env" (
    echo OK - .env already exists
) else (
    copy .env.example .env >nul
    echo OK - Created .env from template
    echo WARNING: Edit .env and add your GROQ_API_KEY
    echo Get free key from: https://console.groq.com/keys
)
echo.

REM 6. Create directories
echo [6/8] Creating necessary directories...
if not exist "chroma_db" mkdir chroma_db
if not exist "logs" mkdir logs
echo OK - Directories created
echo.

REM 7. Test LLM connection
echo [7/8] Testing LLM connection...
python -c "from app.llm_client import LLMClient; llm = LLMClient(); print('OK - LLM client initialized')" 2>nul || echo WARNING - LLM test failed (check GROQ_API_KEY in .env)
echo.

REM 8. Test imports
echo [8/8] Testing imports...
python -c "from app.agents.orchestrator import OrchestratorAgent; from app.evaluation.pipeline import EvaluationPipeline; print('OK - All imports successful')" 2>nul
if errorlevel 1 (
    echo ERROR: Import test failed
    exit /b 1
)
echo.

echo ==================================
echo SETUP COMPLETE!
echo ==================================
echo.
echo Next steps:
echo   1. Edit .env and add your GROQ_API_KEY
echo   2. Start server: python api.py
echo   3. Test API: curl http://localhost:8000/health
echo   4. View docs: http://localhost:8000/docs
echo.
echo Or use Docker:
echo   docker-compose up
echo.
