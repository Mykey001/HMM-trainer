@echo off
echo ============================================
echo  Market Regime Model - Setup & Training
echo ============================================
echo.

echo [1/3] Installing dependencies...
pip install hmmlearn seaborn scipy --quiet
if errorlevel 1 (
    echo [WARNING] Some packages failed. Trying individually...
    pip install seaborn --quiet
    pip install scipy --quiet
    pip install hmmlearn --quiet
)
echo.

echo [2/3] Testing feature engine...
python feature_engine.py
if errorlevel 1 (
    echo [ERROR] Feature engine test failed!
    pause
    exit /b 1
)
echo.

echo [3/3] Running training pipeline...
echo This will take several minutes for 536K bars...
echo.
python train_regime_model.py
echo.

echo ============================================
echo  Training Complete!
echo  Check evaluation_results/ for charts
echo ============================================
pause
