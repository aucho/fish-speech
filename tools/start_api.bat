@echo off
REM Fish Speech API 服务器启动脚本 (Windows)
REM 用法: start_api.bat [选项]

setlocal enabledelayedexpansion

REM 设置默认值
if "%MODE%"=="" set MODE=tts
if "%LLAMA_CHECKPOINT_PATH%"=="" set LLAMA_CHECKPOINT_PATH=checkpoints/openaudio-s1-mini
if "%DECODER_CHECKPOINT_PATH%"=="" set DECODER_CHECKPOINT_PATH=checkpoints/openaudio-s1-mini/codec.pth
if "%DECODER_CONFIG_NAME%"=="" set DECODER_CONFIG_NAME=modded_dac_vq
if "%DEVICE%"=="" set DEVICE=cuda
if "%LISTEN%"=="" set LISTEN=127.0.0.1:8080
if "%WORKERS%"=="" set WORKERS=1
if "%MAX_TEXT_LENGTH%"=="" set MAX_TEXT_LENGTH=0
if "%API_KEY%"=="" set API_KEY=

set HALF_FLAG=
set COMPILE_FLAG=--compile
REM 默认启用编译加速

REM 解析命令行参数
:parse_args
if "%~1"=="" goto :build_cmd
if /i "%~1"=="--mode" (
    set MODE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--llama-checkpoint-path" (
    set LLAMA_CHECKPOINT_PATH=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--decoder-checkpoint-path" (
    set DECODER_CHECKPOINT_PATH=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--decoder-config-name" (
    set DECODER_CONFIG_NAME=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--device" (
    set DEVICE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--listen" (
    set LISTEN=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--workers" (
    set WORKERS=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--max-text-length" (
    set MAX_TEXT_LENGTH=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--api-key" (
    set API_KEY=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--half" (
    set HALF_FLAG=--half
    shift
    goto :parse_args
)
if /i "%~1"=="--compile" (
    set COMPILE_FLAG=--compile
    shift
    goto :parse_args
)
if /i "%~1"=="--no-compile" (
    set COMPILE_FLAG=
    shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    goto :show_help
)
if /i "%~1"=="-h" (
    goto :show_help
)
echo 未知选项: %~1
echo 使用 --help 查看帮助信息
exit /b 1

:show_help
echo Fish Speech API 服务器启动脚本
echo.
echo 用法: %~nx0 [选项]
echo.
echo 选项:
echo   --mode MODE                      运行模式 (默认: tts)
echo   --llama-checkpoint-path PATH     LLaMA 检查点路径 (默认: checkpoints/openaudio-s1-mini)
echo   --decoder-checkpoint-path PATH   解码器检查点路径 (默认: checkpoints/openaudio-s1-mini/codec.pth)
echo   --decoder-config-name NAME       解码器配置名称 (默认: modded_dac_vq)
echo   --device DEVICE                  设备类型 (默认: cuda, 可选: cpu)
echo   --listen ADDRESS:PORT            监听地址和端口 (默认: 127.0.0.1:8080)
echo   --workers NUM                    工作进程数 (默认: 1)
echo   --max-text-length NUM            最大文本长度 (默认: 0, 0表示无限制)
echo   --api-key KEY                    API 密钥 (可选，用于身份验证)
echo   --half                           使用半精度浮点数
echo   --compile                        启用编译模式 (默认启用)
echo   --no-compile                     禁用编译模式
echo   --help, -h                       显示此帮助信息
echo.
echo 环境变量:
echo   也可以通过环境变量设置上述选项，例如:
echo   set DEVICE=cpu
echo   set LISTEN=0.0.0.0:8080
echo   set API_KEY=your_api_key
echo   %~nx0
exit /b 0

:build_cmd
REM 构建命令
set CMD=python tools\api_server.py
set CMD=!CMD! --mode !MODE!
set CMD=!CMD! --llama-checkpoint-path "!LLAMA_CHECKPOINT_PATH!"
set CMD=!CMD! --decoder-checkpoint-path "!DECODER_CHECKPOINT_PATH!"
set CMD=!CMD! --decoder-config-name !DECODER_CONFIG_NAME!
set CMD=!CMD! --device !DEVICE!
set CMD=!CMD! --listen !LISTEN!
set CMD=!CMD! --workers !WORKERS!
set CMD=!CMD! --max-text-length !MAX_TEXT_LENGTH!

if not "!API_KEY!"=="" (
    set CMD=!CMD! --api-key "!API_KEY!"
)

if not "!HALF_FLAG!"=="" (
    set CMD=!CMD! !HALF_FLAG!
)

if not "!COMPILE_FLAG!"=="" (
    set CMD=!CMD! !COMPILE_FLAG!
)

REM 显示启动信息
echo ==========================================
echo 启动 Fish Speech API 服务器
echo ==========================================
echo 模式: !MODE!
echo 设备: !DEVICE!
echo 监听地址: !LISTEN!
echo 工作进程数: !WORKERS!
if not "!COMPILE_FLAG!"=="" (
    echo 编译加速: 启用
) else (
    echo 编译加速: 禁用
)
echo LLaMA 检查点: !LLAMA_CHECKPOINT_PATH!
echo 解码器检查点: !DECODER_CHECKPOINT_PATH!
echo ==========================================
echo.

REM 执行命令
!CMD!

endlocal

