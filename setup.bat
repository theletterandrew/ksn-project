@echo off
echo ==================================================
echo   San Bernardino Ksn: Environment Setup
echo ==================================================

:: 1. Safety Check
if not exist "setup_project.py" (
    echo [ERROR] setup_project.py not found in this directory.
    echo Please 'cd' into your project folder before running this script.
    pause
    exit /b
)

:: 2. Find conda installation
echo [1/3] Locating conda...
call conda --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] conda not found. Make sure Anaconda/Miniconda is installed
    echo and that you are running this from the ArcGIS Pro Python Command Prompt.
    pause
    exit /b
)

:: 3. Create the Conda Environment
echo [2/3] Building ksn_env from environment.yml...
call conda env create -f env/environment.yml
if errorlevel 1 (
    echo [INFO] Environment already exists. Updating instead...
    call conda env update -f env/environment.yml --prune
)

:: 4. Verify environment was created
call conda env list | findstr "ksn_env" >nul
if errorlevel 1 (
    echo [ERROR] ksn_env was not created successfully. Check environment.yml for errors.
    pause
    exit /b
)

:: 5. Run verification script inside ksn_env directly (no activate needed)
echo [3/3] Running project verification...
call conda run -n ksn_env python setup_project.py
if errorlevel 1 (
    echo [ERROR] Verification script failed.
    pause
    exit /b
)

echo.
echo ==================================================
echo Setup Complete!
echo To use this environment in future sessions, run:
echo     conda activate ksn_env
echo ==================================================

:: Open a new prompt with ksn_env already active
cmd /k "conda activate ksn_env"