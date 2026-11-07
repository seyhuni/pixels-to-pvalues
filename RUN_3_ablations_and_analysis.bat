@echo off
REM ===========================================================================
REM 3) Ablations, LOPO, seeds, ensemble, baselines and all statistical analyses.
REM    Usage:  RUN_3_ablations_and_analysis.bat            (backbone chosen on INNER-validation metrics)
REM            RUN_3_ablations_and_analysis.bat fm:uni2h   (force a backbone)
REM    Stage-2..4 run names get the backbone tag appended, e.g. a2_dual_uni2h.
REM    Crash-safe: each plan restarts automatically (up to 100 times) and resumes from the last epoch.
REM ===========================================================================
setlocal
cd /d %~dp0
set BEST=%1
if "%BEST%"=="" for /f "delims=" %%E in ('python scripts\pick_best.py --prefix s1_') do set BEST=%%E
if "%BEST%"=="" goto :nobest
for /f "delims=" %%R in ('python scripts\pick_best.py --prefix s1_ --encoder %BEST% --field run') do set BESTRUN=%%R
for /f "delims=" %%T in ('python scripts\pick_best.py --encoder %BEST% --field tag') do set TAG=%%T
if "%BESTRUN%"=="" goto :nobest
echo Backbone: %BEST%   stage-1 run: %BESTRUN%   tag: %TAG%

call :runplan plans\stage2_ablation.yaml --set model.encoder=%BEST% || goto :err
call :runplan plans\stage3_lopo.yaml --set model.encoder=%BEST% || goto :err
call :runplan plans\stage4_seeds.yaml --set model.encoder=%BEST% || goto :err
python scripts\03_ensemble.py --runs %BESTRUN% e4_seed1_%TAG% e4_seed2_%TAG% --name final_ens_%TAG% || goto :err

REM ---- baselines that do not need nnU-Net
python baselines\classical_deconv.py || goto :err
python baselines\official_unet_fixed.py || echo (official baseline skipped - pip install monai)

REM ---- analyses
python scripts\04_summarize.py --ref a2_dual_%TAG% --B 2000 || goto :err
python scripts\05_conformal.py --exp final_ens_%TAG% || goto :err
python scripts\05_conformal.py --exp %BESTRUN% --splits 300 || goto :err
python scripts\06_ppi.py --exp final_ens_%TAG% || goto :err
python scripts\07_rater_analysis.py --exp final_ens_%TAG% || goto :err
python scripts\08_pack_results.py
echo === STEP 3 DONE - send results_for_paper.zip ===
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
REM kill DataLoader worker processes orphaned by a native crash (they hold RAM/GPU handles)
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'spawn_main' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
timeout /t 60 /nobreak >nul
goto :retry

:nobest
echo Could not determine the backbone / stage-1 run. Run RUN_2_backbones.bat first or pass e.g. fm:uni2h
exit /b 1

:err
echo FAILED with error %errorlevel%
exit /b 1
