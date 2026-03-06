#!/bin/bash

# RT-DETRv2 + DINOv3 训练脚本 (DeepPCB 数据集)
# 使用 DINOv3 ViT-Base 作为骨干网络

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 进入工作目录
cd /home/wh/fj/RT-DETR/rtdetrv2_pytorch

# 创建输出目录
mkdir -p output/rtdetrv2_dinov3b_visdrone

# 设置训练参数
CONFIG_FILE="configs/rtdetrv2/rtdetrv2_dinov3b_visdrone.yml"
OUTPUT_DIR="./output/rtdetrv2_dinov3b_visdrone"
LOG_FILE="$OUTPUT_DIR/train.log"

# 打印训练信息
echo "=========================================="
echo "RT-DETRv2 + DINOv3 ViT-Base (VisDrone)"
echo "=========================================="
echo "Date: $(date)"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# 检查数据集路径
if [ ! -d "/home/wh/fj/Datasets/visdrone" ]; then
    echo "Error: Dataset not found at /home/wh/fj/Datasets/visdrone"
    exit 1
fi

# 启动训练
echo "Starting training with DINOv3 backbone..."
echo "Use 'tail -f $LOG_FILE' to monitor training progress"
echo "Press Ctrl+C to stop training"

# 单GPU训练
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    -c $CONFIG_FILE \
    --use-amp \
    --output-dir $OUTPUT_DIR \
    2>&1 | tee $LOG_FILE

echo "Training completed at $(date)"
