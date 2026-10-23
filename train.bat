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
REM Settings: copy .env.example to .env and edit it (DEAI_RUN, DEAI_REWARD, DEAI_ROUNDS,
REM DEAI_BASE, DEAI_LR, ..., TB_PORT, SLOPTOTAL_DIR, HF_HOME). Variables already set in
REM the environment win over .env.
REM (Loops stay outside parenthesised blocks: cmd breaks labels and %VAR% inside them.)

setlocal
pushd "%~dp0"

REM --- .env: KEY=value lines, # comments; doesn't override the environment ---
if exist ".env" for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do if not defined %%A set "%%A=%%B"

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
REM Optional settings become flags only when set, so the trainer's defaults apply otherwise.
set "ARGS="
if defined DEAI_BASE set "ARGS=%ARGS% --base %DEAI_BASE%"
if defined DEAI_N set "ARGS=%ARGS% --n %DEAI_N%"
if defined DEAI_TEMPERATURE set "ARGS=%ARGS% --temperature %DEAI_TEMPERATURE%"
if defined DEAI_GEN_BATCH set "ARGS=%ARGS% --gen-batch %DEAI_GEN_BATCH%"
if defined DEAI_TRAIN_LIMIT set "ARGS=%ARGS% --train-limit %DEAI_TRAIN_LIMIT%"
if defined DEAI_EVAL_LIMIT set "ARGS=%ARGS% --eval-limit %DEAI_EVAL_LIMIT%"
if defined DEAI_LR set "ARGS=%ARGS% --lr %DEAI_LR%"
if defined DEAI_BETA set "ARGS=%ARGS% --beta %DEAI_BETA%"
if defined DEAI_LORA_R set "ARGS=%ARGS% --lora-r %DEAI_LORA_R%"
if defined DEAI_LOSS_TYPE set "ARGS=%ARGS% --loss-type %DEAI_LOSS_TYPE%"
if defined DEAI_EFFECTIVE_BATCH set "ARGS=%ARGS% --effective-batch %DEAI_EFFECTIVE_BATCH%"
if defined DEAI_EXTRA_ARGS set "ARGS=%ARGS% %DEAI_EXTRA_ARGS%"
set "PYTHONIOENCODING=utf-8"
echo.
echo Training run "%DEAI_RUN%" (reward %DEAI_REWARD%, %DEAI_ROUNDS% rounds). Ctrl+C stops; rerun to resume.
echo.
if defined ARGS echo Settings:%ARGS%
"%PY%" -m deai.train_dpo --run %DEAI_RUN% --reward %DEAI_REWARD% --rounds %DEAI_ROUNDS%%ARGS% %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%

:fail
popd
exit /b 1
