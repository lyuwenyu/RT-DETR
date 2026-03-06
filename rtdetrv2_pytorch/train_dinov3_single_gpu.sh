#!/bin/bash

# RT-DETRv2 Single-GPU Training Script with DINOv3 Backbone
# 使用 DINOv3 ViT-Base 作为骨干网络训练 RT-DETRv2

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/wh/fj/RT-DETR/rtdetrv2_pytorch:$PYTHONPATH"

# 激活conda环境
source /home/wh/anaconda3/bin/activate rtdetrv2-pytorch

# 进入工作目录
cd /home/wh/fj/RT-DETR/rtdetrv2_pytorch

# 创建输出目录
mkdir -p output/rtdetrv2_dinov3b_1x_coco

# 设置训练参数
CONFIG_FILE="configs/rtdetrv2/rtdetrv2_dinov3b_1x_coco.yml"
OUTPUT_DIR="./output/rtdetrv2_dinov3b_1x_coco"
LOG_FILE="$OUTPUT_DIR/train.log"

# 打印训练信息
echo "=========================================="
echo "RT-DETRv2 + DINOv3 ViT-Base Training"
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
if [ ! -d "./dataset/coco" ]; then
    echo "Error: Dataset not found at ./dataset/coco"
    echo "Please download COCO dataset or set correct path in config"
    exit 1
fi

# 启动训练
echo "Starting single-GPU training with DINOv3 backbone..."
echo "Use 'tail -f $LOG_FILE' to monitor training progress"
echo "Press Ctrl+C to stop training"

# 单GPU训练
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    -c $CONFIG_FILE \
    --use-amp \
    --output-dir $OUTPUT_DIR \
    2>&1 | tee $LOG_FILE

echo "Training completed at $(date)"
