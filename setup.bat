@echo off
echo ==================================================
echo   San Bernardino Ksn: Environment Setup
echo ==================================================

:: 1. Safety Check: Verify we are in the right folder
if not exist "setup_project.py" (
    echo [ERROR] setup_project.py not found in this directory.
    echo Please 'cd' into your project folder before running this script.
    pause
    exit /b
)

:: 2. Create the Conda Environment
echo [1/3] Building ksn_env from environment.yml...
call conda env create -f env/environment.yml --quiet
if errorlevel 1 (
    echo [INFO] Environment may already exist, skipping creation...
)

:: 3. Activate the Environment
echo [2/3] Activating ksn_env...
call conda init
call conda activate ksn_env

:: 4. Run Python Verification
echo [3/3] Running project verification...
python setup_project.py

echo.
echo Setup Complete! 
echo You are now working within the 'ksn_env' environment.
echo ==================================================
:: Keeps the prompt open and active in the new environment
cmd /k