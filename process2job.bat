@echo off

REM Ensure an argument is passed
if "%~1"=="" (
    echo Error: No arguments provided. Usage: %~nx0 --[train|test] --[args...]
    exit /b 1
)

REM Get the first argument
set "MODE=%~1"
shift /1

REM Check the mode and call the appropriate Python script
if "%MODE%"=="--train" (
    python train.py %* --out_dir="out\" --data_dir="D:\datasets\"
) else if "%MODE%"=="--test" (
    python test.py %* --out_dir="out\" --data_dir="D:\datasets\"
) else (
    echo Error: Invalid mode '%MODE%'. Use '--train' or '--test'.
    exit /b 1
)
