#!/bin/bash

# RT-DETRv2 Training Monitor Script
# This script monitors the training progress and shows key metrics

# 设置输出目录
OUTPUT_DIR="./output/rtdetrv2_r18vd_120e_pku_market_pcb"
LOG_FILE="$OUTPUT_DIR/train.log"

# 检查日志文件是否存在
if [ ! -f "$LOG_FILE" ]; then
    echo "Training log file not found: $LOG_FILE"
    echo "Please start training first using: ./train_pku_market_pcb.sh"
    exit 1
fi

# 获取训练状态
get_training_status() {
    if grep -q "Training completed" "$LOG_FILE"; then
        echo "✅ Training completed"
    elif grep -q "Traceback" "$LOG_FILE"; then
        echo "❌ Training failed with error"
    elif pgrep -f "tools/train.py" > /dev/null; then
        echo "🔄 Training in progress"
    else
        echo "⏸️ Training paused or stopped"
    fi
}

# 显示最新的训练进度
show_latest_progress() {
    echo "=========================================="
    echo "Latest Training Progress"
    echo "=========================================="
    
    # 获取最后20行日志
    tail -n 20 "$LOG_FILE" | grep -E "(Epoch|loss|AP|bbox|accuracy)" || \
    tail -n 10 "$LOG_FILE"
    
    echo ""
}

# 显示统计信息
show_statistics() {
    echo "=========================================="
    echo "Training Statistics"
    echo "=========================================="
    
    # 计算训练时长
    if grep -q "Start training" "$LOG_FILE"; then
        start_time=$(grep "Start training" "$LOG_FILE" | head -1 | cut -d' ' -f1-3)
        echo "Started: $start_time"
    fi
    
    # 显示当前epoch
    if grep -q "Epoch:" "$LOG_FILE"; then
        current_epoch=$(grep "Epoch:" "$LOG_FILE" | tail -1 | grep -o "Epoch: \[.*\]" | head -1)
        echo "Current: $current_epoch"
    fi
    
    # 显示输出目录大小
    if [ -d "$OUTPUT_DIR" ]; then
        dir_size=$(du -sh "$OUTPUT_DIR" | cut -f1)
        echo "Output size: $dir_size"
    fi
    
    # 显示检查点文件
    if ls "$OUTPUT_DIR"/*.pth 1> /dev/null 2>&1; then
        echo "Checkpoints:"
        ls -lh "$OUTPUT_DIR"/*.pth | awk '{print $9 " (" $5 ")"}'
    fi
    
    echo ""
}

# 实时监控模式
if [ "$1" = "--watch" ]; then
    echo "Starting real-time monitoring... Press Ctrl+C to stop"
    while true; do
        clear
        echo "RT-DETRv2 Training Monitor - $(date)"
        echo "=========================================="
        get_training_status
        echo ""
        show_latest_progress
        show_statistics
        echo "Last updated: $(date)"
        echo "=========================================="
        sleep 10
    done
else
    # 单次显示
    get_training_status
    echo ""
    show_latest_progress
    show_statistics
fi