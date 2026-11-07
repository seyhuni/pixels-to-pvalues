@echo off
REM ===========================================================================
REM 4) Complementary runs requested after the analysis of RUN_3 (about 1.5-2 days):
REM    a) published UNet baseline (as published vs corrected)        ~1-2 h
REM    b) leave-one-project-out WITHOUT stain physics (test of B)   ~14 h
REM    c) nnU-Net v2 2d, expert-1 labels, 250 epochs, 5 folds        ~15-25 h
REM    then summary tables + results_for_paper.zip
REM    Crash-safe and resume-safe; re-run the same file after an interruption.
REM ===========================================================================
setlocal
cd /d %~dp0

echo ### a) published baseline
if not exist results\official_unet_corrected\oof_predictions.csv (
  python baselines\official_unet_fixed.py --workers 0 > results\official_unet.log 2>&1
  if errorlevel 1 echo    official baseline FAILED - see results\official_unet.log
)

echo ### b) LOPO without stain physics
call :runplan plans\stage3_lopo.yaml --set model.encoder=fm:uni2h --only l3_stain_none || goto :err

echo ### c) nnU-Net (expert 1, 250 epochs)
set nnUNet_raw=%CD%\nnunet_data\nnUNet_raw
set nnUNet_preprocessed=%CD%\nnunet_data\nnUNet_preprocessed
set nnUNet_results=%CD%\nnunet_data\nnUNet_results
set nnUNet_compile=f
set nnUNet_n_proc_DA=6
set NNUNET_TRAINER_DIR=nnUNetTrainer_250epochs__nnUNetPlans__2d
if not exist nnunet_data\nnUNet_preprocessed\Dataset501_FRPVE1\nnUNetPlans.json (
  python baselines\nnunet\convert.py || goto :err
  nnUNetv2_plan_and_preprocess -d 501 -c 2d --verify_dataset_integrity || goto :err
  python baselines\nnunet\convert.py --splits_only || goto :err
)
for %%F in (0 1 2 3 4) do call :nnfold %%F
python baselines\nnunet\collect.py > results\nnunet_collect.log 2>&1

echo ### summary
python scripts\04_summarize.py --ref a2_dual_uni2h --B 2000
python scripts\08_pack_results.py
echo === STEP 4 DONE ===
goto :eof

:nnfold
set /a NT=0
:nnretry
if exist nnunet_data\nnUNet_results\Dataset501_FRPVE1\%NNUNET_TRAINER_DIR%\fold_%1\checkpoint_final.pth exit /b 0
nnUNetv2_train 501 2d %1 -tr nnUNetTrainer_250epochs --npz --c
if "%errorlevel%"=="0" exit /b 0
set /a NT+=1
echo %date% %time% nnunet fold=%1 restart=%NT% >> results\restarts.log
if %NT% geq 30 exit /b 1
timeout /t 60 /nobreak >nul
goto :nnretry

:runplan
set /a TRIES=0
:retry
python scripts\run_plan.py %*
set RC=%errorlevel%
if "%RC%"=="0" exit /b 0
set /a TRIES+=1
echo %date% %time% plan=%1 rc=%RC% restart=%TRIES% >> results\restarts.log
if %TRIES% geq 100 exit /b 1
echo !!! run_plan stopped (code %RC%) - automatic restart %TRIES%/100 in 60 s
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'spawn_main' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
timeout /t 60 /nobreak >nul
goto :retry

:err
echo FAILED with error %errorlevel%
exit /b 1
