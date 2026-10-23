@echo off
REM DeAI - start Phase 4A training (best-of-N + DPO) and open TensorBoard.
REM
REM Usage:  train.bat [extra train_dpo arguments]
REM   train.bat                                   Qwen3-4B, quick-score reward, 3 rounds
REM   train.bat --reward neural-quick --run r-neural-quick-qwen3-4b
REM   train.bat --rounds 1 --train-limit 8 --eval-limit 8 --n 4 --run smoke    (smoke test)
REM   train.bat --base Qwen/Qwen2.5-1.5B-Instruct --gen-batch 64 --run r-quick
REM
REM Later arguments win, so anything passed here overrides the defaults below.
REM Rerunning the same command resumes an interrupted run.
REM
REM Environment overrides: DEAI_RUN, DEAI_REWARD, DEAI_ROUNDS, TB_PORT, SLOPTOTAL_DIR
REM (Loops stay outside parenthesised blocks: cmd breaks labels and %VAR% inside them.)

setlocal
pushd "%~dp0"

if not defined DEAI_RUN set "DEAI_RUN=r-quick-qwen3-4b"
if not defined DEAI_REWARD set "DEAI_REWARD=quick"
if not defined DEAI_ROUNDS set "DEAI_ROUNDS=3"
if not defined TB_PORT set "TB_PORT=6006"
if not defined SLOPTOTAL_DIR set "SLOPTOTAL_DIR=D:\AI\sloptotal"
set "PY=.venv\Scripts\python.exe"
set "TB_URL=http://localhost:%TB_PORT%/"

if not exist "%PY%" (
    echo Error: %PY% not found. Create the venv first, see README.md "Setup".
    goto :fail
)

REM --- SlopTotal: the detector server must be up before scoring -------------
curl -sf http://localhost:8000/health >nul 2>&1
if not errorlevel 1 goto :sloptotal_up
if not exist "%SLOPTOTAL_DIR%\scripts\start.bat" (
    echo Error: SlopTotal is not running on http://localhost:8000 and
    echo %SLOPTOTAL_DIR%\scripts\start.bat was not found. Start it, or set SLOPTOTAL_DIR.
    goto :fail
)
echo Starting SlopTotal in a new window...
start "SlopTotal" /D "%SLOPTOTAL_DIR%" cmd /k scripts\start.bat
set /a WAITED=0
:wait_sloptotal
timeout /t 2 /nobreak >nul
set /a WAITED+=2
curl -sf http://localhost:8000/health >nul 2>&1
if not errorlevel 1 goto :sloptotal_up
if %WAITED% LSS 300 goto :wait_sloptotal
echo Error: SlopTotal did not answer within 5 minutes. Check its window.
goto :fail
:sloptotal_up
echo SlopTotal is up.

REM --- TensorBoard: start once, reuse if already running --------------------
curl -sf %TB_URL% >nul 2>&1
if not errorlevel 1 goto :tb_up
echo Starting TensorBoard on port %TB_PORT%...
start "TensorBoard" /min "%PY%" -m tensorboard.main --logdir runs --port %TB_PORT% --reload_interval 15
set /a WAITED=0
:wait_tb
timeout /t 1 /nobreak >nul
set /a WAITED+=1
curl -sf %TB_URL% >nul 2>&1
if not errorlevel 1 goto :tb_up
if %WAITED% LSS 60 goto :wait_tb
echo Warning: TensorBoard did not start; training continues without it.
goto :train
:tb_up
echo TensorBoard: %TB_URL%
start "" %TB_URL%

REM --- Training ---------------------------------------------------------------
:train
set "PYTHONIOENCODING=utf-8"
echo.
echo Training run "%DEAI_RUN%" (reward %DEAI_REWARD%, %DEAI_ROUNDS% rounds). Ctrl+C stops; rerun to resume.
echo.
"%PY%" -m deai.train_dpo --run %DEAI_RUN% --reward %DEAI_REWARD% --rounds %DEAI_ROUNDS% %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%

:fail
popd
exit /b 1
