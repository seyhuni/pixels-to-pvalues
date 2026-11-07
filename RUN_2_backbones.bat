@echo off
REM ===========================================================================
REM 2) Stage 1 - backbone comparison, official 5-fold CV (long)
REM    Resume-safe: after a crash the plan restarts automatically (up to 100 times)
REM    and continues from the last finished epoch of the interrupted fold.
REM ===========================================================================
setlocal
cd /d %~dp0
call :runplan plans\stage1_backbones.yaml || goto :err
python scripts\04_summarize.py || goto :err
python scripts\pick_best.py --prefix s1_ --verbose
echo === STEP 2 DONE ===
goto :eof

:runplan
set /a TRIES=0
:retry
python scripts\run_plan.py %*
set RC=%errorlevel%
REM a native crash (e.g. access violation) gives a NEGATIVE code -> test for exactly 0
if "%RC%"=="0" exit /b 0
set /a TRIES+=1
if not exist results mkdir results
echo %date% %time% plan=%1 rc=%RC% restart=%TRIES% >> results\restarts.log
if %TRIES% geq 100 exit /b 1
echo.
echo !!! run_plan stopped (code %RC%) - automatic restart %TRIES%/100 in 60 s (Ctrl+C to cancel)
timeout /t 60 /nobreak >nul
goto :retry

:err
echo FAILED with error %errorlevel%
exit /b 1
