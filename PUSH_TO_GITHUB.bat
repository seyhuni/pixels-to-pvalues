@echo off
REM ===========================================================================
REM Push the code to https://github.com/seyhuni/pixels-to-pvalues (private repo).
REM 1) First create an EMPTY repo on github.com/new:
REM       name: pixels-to-pvalues   visibility: Private   (no README, no license, no .gitignore)
REM 2) Double-click this file (or run it from Anaconda Prompt).
REM    Data, cache, results, checkpoints and the manuscript are excluded by .gitignore.
REM    The first push opens a browser window for GitHub sign-in (Git Credential Manager).
REM Running it again later commits and pushes any new changes.
REM ===========================================================================
setlocal
cd /d %~dp0
set REPO=https://github.com/seyhuni/pixels-to-pvalues.git

where git >nul 2>nul
if errorlevel 1 (
  echo Git was not found. Install it from https://git-scm.com/download/win and run this file again.
  pause
  exit /b 1
)

if not exist .git (
  git init
  git checkout -b main
)

git config user.name >nul 2>nul || git config user.name "Huseyin Yanik"
git config user.email >nul 2>nul || git config user.email "huseyinyanik@mersin.edu.tr"

git add -A
echo.
echo Files to be committed:
git status --short
echo.
git diff --cached --quiet
if errorlevel 1 (
  git commit -q -m "Code for 'From pixels to p-values' (MLST submission)" -m "Rater-aware, hierarchy-consistent, stain-physics-informed segmentation of FRPV-PH; animal-level conformal intervals; prediction-powered inference." -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01FdWterv74v4DETBBZhw4rr"
) else (
  echo Nothing new to commit.
)

git remote get-url origin >nul 2>nul
if errorlevel 1 (git remote add origin %REPO%) else (git remote set-url origin %REPO%)

git push -u origin main
if errorlevel 1 (
  echo.
  echo Push failed. Check that the empty repo exists at %REPO% and that you are signed in to GitHub.
) else (
  echo.
  echo Done: https://github.com/seyhuni/pixels-to-pvalues
)
pause
