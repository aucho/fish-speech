#!/bin/bash

# Fish Speech API 服务器启动脚本
# 用法: ./start_api.sh [选项]

# 激活 conda 环境
CONDA_ENV="fish-speech"

# 尝试初始化 conda（支持多个常见安装位置）
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
    source "$CONDA_PREFIX/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    # 如果 conda 已在 PATH 中，尝试直接使用
    eval "$(conda shell.bash hook)"
else
    echo "警告: 未找到 conda，尝试直接运行（如果已激活环境可忽略此警告）"
fi

# 激活 fish-speech 环境
if command -v conda &> /dev/null; then
    echo "正在激活 conda 环境: $CONDA_ENV"
    conda activate "$CONDA_ENV" || {
        echo "错误: 无法激活 conda 环境 '$CONDA_ENV'"
        echo "请确保已创建该环境: conda create -n $CONDA_ENV python=3.10"
        exit 1
    }
    echo "已激活 conda 环境: $CONDA_ENV"
    echo ""
else
    echo "警告: conda 命令不可用，假设环境已激活"
    echo ""
fi

# 设置默认值
MODE=${MODE:-tts}
LLAMA_CHECKPOINT_PATH=${LLAMA_CHECKPOINT_PATH:-checkpoints/openaudio-s1-mini}
DECODER_CHECKPOINT_PATH=${DECODER_CHECKPOINT_PATH:-checkpoints/openaudio-s1-mini/codec.pth}
DECODER_CONFIG_NAME=${DECODER_CONFIG_NAME:-modded_dac_vq}
DEVICE=${DEVICE:-cuda}
LISTEN=${LISTEN:-127.0.0.1:8080}
WORKERS=${WORKERS:-1}
MAX_TEXT_LENGTH=${MAX_TEXT_LENGTH:-0}
API_KEY=${API_KEY:-}

# 解析命令行参数
HALF_FLAG=""
COMPILE_FLAG="--compile"  # 默认启用编译加速

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --llama-checkpoint-path)
            LLAMA_CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --decoder-checkpoint-path)
            DECODER_CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --decoder-config-name)
            DECODER_CONFIG_NAME="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --listen)
            LISTEN="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --max-text-length)
            MAX_TEXT_LENGTH="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        --half)
            HALF_FLAG="--half"
            shift
            ;;
        --compile)
            COMPILE_FLAG="--compile"
            shift
            ;;
        --no-compile)
            COMPILE_FLAG=""
            shift
            ;;
        --help|-h)
            echo "Fish Speech API 服务器启动脚本"
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --mode MODE                      运行模式 (默认: tts)"
            echo "  --llama-checkpoint-path PATH     LLaMA 检查点路径 (默认: checkpoints/openaudio-s1-mini)"
            echo "  --decoder-checkpoint-path PATH   解码器检查点路径 (默认: checkpoints/openaudio-s1-mini/codec.pth)"
            echo "  --decoder-config-name NAME       解码器配置名称 (默认: modded_dac_vq)"
            echo "  --device DEVICE                  设备类型 (默认: cuda, 可选: cpu)"
            echo "  --listen ADDRESS:PORT            监听地址和端口 (默认: 127.0.0.1:8080)"
            echo "  --workers NUM                    工作进程数 (默认: 1)"
            echo "  --max-text-length NUM            最大文本长度 (默认: 0, 0表示无限制)"
            echo "  --api-key KEY                    API 密钥 (可选，用于身份验证)"
            echo "  --half                           使用半精度浮点数"
            echo "  --compile                        启用编译模式 (默认启用)"
            echo "  --no-compile                     禁用编译模式"
            echo "  --help, -h                       显示此帮助信息"
            echo ""
            echo "环境变量:"
            echo "  也可以通过环境变量设置上述选项，例如:"
            echo "  export DEVICE=cpu"
            echo "  export LISTEN=0.0.0.0:8080"
            echo "  export API_KEY=your_api_key"
            echo "  $0"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 构建命令
CMD="python tools/api_server.py"
CMD="$CMD --mode $MODE"
CMD="$CMD --llama-checkpoint-path \"$LLAMA_CHECKPOINT_PATH\""
CMD="$CMD --decoder-checkpoint-path \"$DECODER_CHECKPOINT_PATH\""
CMD="$CMD --decoder-config-name $DECODER_CONFIG_NAME"
CMD="$CMD --device $DEVICE"
CMD="$CMD --listen $LISTEN"
CMD="$CMD --workers $WORKERS"
CMD="$CMD --max-text-length $MAX_TEXT_LENGTH"

if [ -n "$API_KEY" ]; then
    CMD="$CMD --api-key \"$API_KEY\""
fi

if [ -n "$HALF_FLAG" ]; then
    CMD="$CMD $HALF_FLAG"
fi

if [ -n "$COMPILE_FLAG" ]; then
    CMD="$CMD $COMPILE_FLAG"
fi

# 显示启动信息
echo "=========================================="
echo "启动 Fish Speech API 服务器"
echo "=========================================="
echo "模式: $MODE"
echo "设备: $DEVICE"
echo "监听地址: $LISTEN"
echo "工作进程数: $WORKERS"
if [ -n "$COMPILE_FLAG" ]; then
    echo "编译加速: 启用"
else
    echo "编译加速: 禁用"
fi
echo "LLaMA 检查点: $LLAMA_CHECKPOINT_PATH"
echo "解码器检查点: $DECODER_CHECKPOINT_PATH"
echo "=========================================="
echo ""

# 执行命令
eval $CMD

