# PKU-Market-PCB YOLO to COCO Format Conversion

## 概述

本工具将PKU-Market-PCB数据集中的YOLO格式标注转换为RT-DETRv2所需的COCO JSON格式。

## 数据集统计

转换后的数据集统计信息：

### 训练集 (483 images, 2046 annotations)
- missing_hole: 335
- mouse_bite: 344
- open_circuit: 335
- short: 342
- spur: 336
- spurious_copper: 354

### 验证集 (138 images, 602 annotations)
- missing_hole: 108
- mouse_bite: 98
- open_circuit: 100
- short: 97
- spur: 102
- spurious_copper: 97

### 测试集 (72 images, 305 annotations)
- missing_hole: 54
- mouse_bite: 50
- open_circuit: 47
- short: 52
- spur: 50
- spurious_copper: 52

## 图像尺寸分布

数据集包含10种不同的PCB尺寸：

- 2240x2016: 22 images
- 2282x2248: 40 images
- 2529x2530: 36 images
- 2544x2156: 40 images
- 2759x2154: 37 images
- 2775x2159: 44 images
- 2868x2316: 44 images
- 2904x1921: 44 images
- 3034x1586: 91 images
- 3056x2464: 85 images

## 生成的文件

转换后会在数据集目录中生成以下文件：

- `instances_train2017.json` - 训练集标注
- `instances_val2017.json` - 验证集标注
- `instances_test2017.json` - 测试集标注

## 使用方法

### 1. 转换数据集

```bash
python yolo_to_coco.py --dataset_dir /path/to/PKU-Market-PCB --output_dir /path/to/output
```

### 2. 验证转换结果

```bash
python validate_coco.py --data_dir /path/to/PKU-Market-PCB
```

### 3. 获取PCB尺寸信息

```bash
python get_pcb_sizes.py
```

## 转换过程说明

### YOLO格式
- 格式：`class_id x_center y_center width height`
- 坐标：归一化值 (0-1)
- 原点：图像中心

### COCO格式
- 格式：`[x_min, y_min, width, height]`
- 坐标：绝对像素值
- 原点：图像左上角

### 尺寸处理
转换脚本会根据图像文件名的前两位数字（如 `01_missing_hole_02.jpg` 中的 `01`）来识别对应的PCB模板，并使用该模板的实际尺寸进行坐标转换。

## 类别映射

| 类别ID | 类别名称 |
|--------|----------|
| 0 | missing_hole |
| 1 | mouse_bite |
| 2 | open_circuit |
| 3 | short |
| 4 | spur |
| 5 | spurious_copper |

## 注意事项

1. **尺寸识别**：脚本假设图像文件名以PCB ID开头（如 `01_missing_hole_02.jpg`）
2. **文件存在性**：转换前确保所有图像文件都存在
3. **标注格式**：YOLO文件每行一个标注，包含5个数值
4. **坐标范围**：YOLO坐标必须在0-1范围内

## 验证结果

所有生成的JSON文件都通过了完整的COCO格式验证，包括：
- 必要字段检查
- 类别ID验证
- 图像ID唯一性检查
- 标注ID唯一性检查
- 边界框格式验证
- 坐标范围验证

转换后的JSON文件可以直接用于RT-DETRv2训练。