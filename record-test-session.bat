@echo off
setlocal enabledelayedexpansion
title Discovery Copilot -- transcript test session

REM One-button transcript test session (no coach, no phone surface).
REM
REM WARNING: never run two of these at once. Two PyAudio instances in the same
REM machine is an uncatchable Windows access violation that kills the process --
REM starting a second recording would destroy the first one's session.

REM %~dp0 is this script's own directory, so the file works from any checkout.
REM It used to hardcode one machine's path, which is a silent failure on
REM anyone else's clone: `cd /d` to a directory that does not exist leaves the
REM working directory wherever it was, and the venv lookup below then fails
REM for a reason that has nothing to do with the venv.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
  echo ERROR: %PY% not found. A bare "python" is a system 3.12 with no pytest
  echo and will silently do the wrong thing -- fix the venv before recording.
  pause
  exit /b 1
)

echo ============================================================
echo   PRE-FLIGHT -- these cannot be fixed in code
echo ============================================================
echo   * Set the call app's output device NOW. Do NOT change it mid-call.
echo   * Close anything else playing audio -- loopback captures EVERYTHING.
echo   * Set the power plan to never sleep for the call's duration.
echo.
echo   Press any key to run the 5-second precheck, or close this window.
pause >nul

echo.
echo --- precheck (play some audio AND speak into the mic now) ---
"%PY%" -u -m app precheck --seconds 5
echo.
echo   A FAILED microphone row is usually just "you were not talking loudly
echo   enough during those 5 seconds" -- it is not necessarily broken.
echo.
echo   Press any key to START RECORDING, or close this window to abort.
pause >nul

echo.
echo ============================================================
echo   RECORDING. Press Ctrl+C ONCE to stop.
echo   Then answer N to "Terminate batch job?" so the transcript renders.
echo ============================================================
echo.
"%PY%" -u -m app start --retain-audio --no-serve

echo.
echo --- rendering the session just recorded ---
set "LATEST="
for /f "delims=" %%i in ('dir /b /ad /o-d sessions 2^>nul') do (
  if not defined LATEST set "LATEST=%%i"
)
if not defined LATEST (
  echo ERROR: no session directory found under sessions\
  pause
  exit /b 1
)

"%PY%" -u -m app render "sessions\!LATEST!"
echo.
echo ============================================================
echo   Session:    sessions\!LATEST!
echo   Transcript: sessions\!LATEST!\transcript.md
echo ============================================================
echo.
echo Opening the transcript...
start "" "sessions\!LATEST!\transcript.md"
pause
