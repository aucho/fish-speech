@echo off
REM 激活 fish-speech conda 环境并启动 API 服务器

REM 初始化 conda（如果需要）
call conda activate fish-speech

REM 运行 API 服务器
python -m tools.api_server ^
    --listen 0.0.0.0:8080 ^
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" ^
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" ^
    --decoder-config-name modded_dac_vq ^
    --compile

pause