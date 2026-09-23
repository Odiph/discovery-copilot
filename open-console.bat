@echo off
setlocal enabledelayedexpansion
title Discovery Copilot -- console

REM Opens the local console: pick a project, see its survey, hypotheses, fact
REM count and past interviews, then press Start.
REM
REM The console and the interview are ONE process. Pressing Start stops the
REM page's server and this same window then runs the interview -- so leave this
REM window open, and read it, once you have pressed Start. Ctrl+C here before
REM choosing cancels without starting anything.
REM
REM WARNING: never run two interviews at once. Two PyAudio instances on one
REM machine is an uncatchable Windows access violation that kills the process --
REM starting a second one would destroy the first one's session.

REM Workspace-relative: %~dp0 is the directory THIS file lives in, so the
REM script works from any checkout on any machine. It used to name a drive
REM letter, which made it work only on its author's laptop.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
  echo ERROR: %PY% not found. A bare "python" is a system 3.12 with no pytest
  echo and will silently do the wrong thing -- fix the venv before starting.
  echo.
  echo In VS Code run the task "Setup: create venv and install", or from this
  echo directory:
  echo   py -3 -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)

"%PY%" -u -m app console
echo.
echo Console exited. Press any key to close this window.
pause >nul
