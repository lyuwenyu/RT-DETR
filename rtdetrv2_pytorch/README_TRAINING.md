# RT-DETRv2 训练脚本使用说明

## 脚本列表

1. **train_pku_market_pcb.sh** - 多GPU训练脚本
2. **train_pku_market_pcb_single_gpu.sh** - 单GPU训练脚本
3. **monitor_training.sh** - 训练监控脚本

## 使用方法

### 1. 多GPU训练（推荐）
```bash
./train_pku_market_pcb.sh
```

### 2. 单GPU训练
```bash
./train_pku_market_pcb_single_gpu.sh
```

### 3. 监控训练进度
```bash
# 单次查看
./monitor_training.sh

# 实时监控（每10秒更新）
./monitor_training.sh --watch
```

### 4. 查看训练日志
```bash
tail -f output/rtdetrv2_r18vd_120e_pku_market_pcb/train.log
```

## 训练参数

- **模型**: RT-DETRv2 with ResNet18
- **数据集**: PKU-Market-PCB (6个缺陷类别)
- **输入尺寸**: 640x640
- **Epochs**: 120
- **批大小**: 16 (每GPU)
- **学习率**: 0.0001
- **优化器**: AdamW
- **混合精度训练**: 启用

## 输出目录

训练完成后，模型和日志将保存在：
```
output/rtdetrv2_r18vd_120e_pku_market_pcb/
├── train.log          # 训练日志
├── best.pth           # 最佳模型
├── last.pth           # 最后一个epoch的模型
└── epoch_*.pth        # 每个epoch的检查点
```

## 故障排除

### 1. CUDA内存不足
如果遇到内存不足错误，请使用单GPU训练脚本。

### 2. 数据集路径错误
确保数据集位于 `/home/wh/fj/Datasets/PKU-Market-PCB/`

### 3. 环境问题
确保已激活正确的conda环境：
```bash
source /home/wh/anaconda3/bin/activate rtdetrv2-pytorch
```

### 4. 权限问题
确保所有脚本都有执行权限：
```bash
chmod +x *.sh
```

## 性能监控

训练过程中可以通过以下命令监控GPU使用情况：
```bash
# 监控GPU使用情况
watch -n 1 nvidia-smi

# 监控CPU和内存使用
htop
```

## 停止训练

要停止训练，按 `Ctrl+C` 或使用以下命令：
```bash
pkill -f "tools/train.py"
```