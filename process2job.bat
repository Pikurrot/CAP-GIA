@echo off

REM Define variables
set "OUT_DIR=out\"
set "DATA_DIR=D:\datasets\"

python train.py %* --out_dir="%OUT_DIR%" --data_dir="%DATA_DIR%"
