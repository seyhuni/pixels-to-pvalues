@echo off
REM ===========================================================================
REM 1) Environment check, data preparation, unit tests, GPU smoke test (~15-30 min)
REM    Run from the MLST_New folder inside the activated conda env:
REM        conda activate mlst
REM        RUN_1_setup_and_smoke.bat
REM ===========================================================================
setlocal
cd /d %~dp0
python scripts\00_check_env.py --hf || goto :err
python scripts\01_prepare_data.py || goto :err
python tests\test_models.py || goto :err
python scripts\07_rater_analysis.py --experts_only || goto :err
python scripts\run_plan.py plans\stage0_smoke.yaml || goto :err
python scripts\04_summarize.py --B 200 || goto :err
echo.
echo === STEP 1 DONE. Check results\s0_*\fold0\train_log.csv and results\summary\summary.md ===
goto :eof
:err
echo FAILED with error %errorlevel%
exit /b 1
