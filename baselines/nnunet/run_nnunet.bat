@echo off
REM ============================================================================
REM nnU-Net v2 baseline (2d, one model per expert, official animal-level folds)
REM Run from the repository root:  baselines\nnunet\run_nnunet.bat
REM Requires: conda activate mlst  +  pip install -r requirements-baselines.txt
REM --c resumes interrupted folds (a finished fold is only re-validated).
REM Expected time: ~5 folds x 2 experts x (1000 epochs default) - long! For a faster
REM baseline add  -tr nnUNetTrainer_250epochs  to the train commands below.
REM ============================================================================
setlocal
set ROOT=%~dp0..\..
set nnUNet_raw=%ROOT%\nnunet_data\nnUNet_raw
set nnUNet_preprocessed=%ROOT%\nnunet_data\nnUNet_preprocessed
set nnUNet_results=%ROOT%\nnunet_data\nnUNet_results
set nnUNet_compile=f
set nnUNet_n_proc_DA=8

python "%ROOT%\baselines\nnunet\convert.py" || goto :err
nnUNetv2_plan_and_preprocess -d 501 502 -c 2d --verify_dataset_integrity || goto :err
python "%ROOT%\baselines\nnunet\convert.py" --splits_only || goto :err

for %%D in (501 502) do (
  for %%F in (0 1 2 3 4) do (
    nnUNetv2_train %%D 2d %%F --npz --c || goto :err
  )
)
python "%ROOT%\baselines\nnunet\collect.py" || goto :err
echo nnU-Net baseline finished.
goto :eof
:err
echo nnU-Net step failed with error %errorlevel%
exit /b 1
