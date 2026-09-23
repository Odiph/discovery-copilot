@echo off
setlocal enabledelayedexpansion
title Discovery Copilot -- overlay

REM One-button launch of the floating record overlay: translucent, always on
REM top, picks a project and runs the interview.
REM
REM This window is MINIMIZED, never hidden, and never launched with
REM pythonw.exe. That is not cosmetic. The overlay stops an interview by
REM sending CTRL_BREAK_EVENT to the interview's process group with
REM GenerateConsoleCtrlEvent, and that API requires the calling process to OWN
REM A CONSOLE. Under pythonw.exe there is no console, the call fails, and the
REM STOP button silently stops stopping the interview -- which means no
REM closed_at, and a transcript truncated wherever it happened to be. So the
REM console stays alive; it just gets out of the way.
REM
REM It is also a real log: if the overlay misbehaves, restore this window and
REM read it.
REM
REM WARNING: never run two interviews at once. Two PyAudio instances on one
REM machine is an uncatchable Windows access violation that kills the process
REM -- starting a second one would destroy the first one's session.

REM Relaunch ourselves minimized, once. The /min argument is the re-entry
REM marker, so this cannot loop.
if not "%~1"=="/min" (
  start "Discovery Copilot -- overlay" /min "%~f0" /min
  exit /b 0
)

REM Workspace-relative: %~dp0 is the directory THIS file lives in, so the
REM script works from any checkout on any machine. Never a drive letter.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PY=.venv\Scripts\python.exe

if not exist "%PY%" (
  echo ERROR: %PY% not found. A bare "python" is a system interpreter with no
  echo pytest and none of this project's pinned packages, and will silently do
  echo the wrong thing.
  echo.
  echo Create the venv first -- in VS Code run the task
  echo   "Setup: create venv and install"
  echo or from this directory:
  echo   py -3 -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)

"%PY%" -u -m app overlay
echo.
echo Overlay exited. Press any key to close this window.
pause >nul
