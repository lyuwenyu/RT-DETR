#!/usr/bin/env python3
"""
PKU-Market-PCB YOLO to COCO Format Converter
将YOLO格式的标注转换为COCO JSON格式
"""

import os
import json
import argparse
from collections import defaultdict
from pathlib import Path
from PIL import Image

# PCB模板尺寸映射 (PCB ID -> (width, height))
PCB_SIZES = {
    '01': (3034, 1586),
    '04': (3056, 2464),
    '05': (2544, 2156),
    '06': (2868, 2316),
    '07': (2904, 1921),
    '08': (2759, 2154),
    '09': (2775, 2159),
    '10': (2240, 2016),
    '11': (2282, 2248),
    '12': (2529, 2530)
}

# 类别映射
CLASS_NAMES = [
    'missing_hole',
    'mouse_bite', 
    'open_circuit',
    'short',
    'spur',
    'spurious_copper'
]

def get_image_size(image_filename):
    """根据图像文件名获取图像尺寸"""
    # 文件名格式: 01_missing_hole_02.jpg -> PCB ID是 '01'
    pcb_id = image_filename.split('_')[0]
    return PCB_SIZES.get(pcb_id, (3034, 1586))  # 默认尺寸

def yolo_to_coco(x_center, y_center, width, height, img_width, img_height):
    """将YOLO格式转换为COCO格式"""
    # YOLO格式是归一化的中心坐标和宽高
    # COCO格式是绝对坐标的左上角和宽高
    
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    # 转换为左上角坐标
    x_min = x_center_abs - width_abs / 2
    y_min = y_center_abs - height_abs / 2
    
    return [x_min, y_min, width_abs, height_abs]

def convert_dataset(yolo_dir, output_json, dataset_type="train"):
    """转换整个数据集"""
    
    # 创建COCO格式数据结构
    coco_data = {
        "info": {
            "description": f"PKU-Market-PCB {dataset_type} dataset",
            "url": "https://github.com/lyuwenyu/RT-DETR",
            "version": "1.0",
            "year": 2024,
            "contributor": "RT-DETR",
            "date_created": "2024/09/11"
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # 添加类别信息
    for idx, class_name in enumerate(CLASS_NAMES):
        coco_data["categories"].append({
            "id": idx,
            "name": class_name,
            "supercategory": "pcb_defect"
        })
    
    # 获取标签文件路径
    labels_dir = os.path.join(yolo_dir, "labels")
    images_dir = os.path.join(yolo_dir, "images")
    
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found: {labels_dir}")
        return
    
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    # 收集所有标签文件
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    # 用于分配唯一ID
    image_id = 0
    annotation_id = 0
    
    print(f"Converting {len(label_files)} files...")
    
    for label_file in sorted(label_files):
        # 获取对应的图像文件
        image_name = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # 获取图像尺寸
        img_width, img_height = get_image_size(image_name)
        
        # 添加图像信息
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": img_width,
            "height": img_height
        })
        
        # 读取YOLO标注文件
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 处理每个标注
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # YOLO格式: class_id x_center y_center width height
            parts = line.split()
            if len(parts) != 5:
                print(f"Warning: Invalid annotation format in {label_file}: {line}")
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 验证类别ID
                if class_id >= len(CLASS_NAMES):
                    print(f"Warning: Invalid class_id {class_id} in {label_file}")
                    continue
                
                # 转换为COCO格式
                bbox = yolo_to_coco(x_center, y_center, width, height, img_width, img_height)
                
                # 计算面积
                area = bbox[2] * bbox[3]
                
                # 添加标注
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0
                })
                
                annotation_id += 1
                
            except ValueError as e:
                print(f"Warning: Error parsing line in {label_file}: {line}")
                continue
        
        image_id += 1
        
        if image_id % 100 == 0:
            print(f"Processed {image_id} images...")
    
    # 保存JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"Conversion completed!")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    
    # 统计每个类别的数量
    category_stats = defaultdict(int)
    for ann in coco_data['annotations']:
        category_stats[ann['category_id']] += 1
    
    print("\nCategory statistics:")
    for class_id, count in sorted(category_stats.items()):
        print(f"  {CLASS_NAMES[class_id]}: {count}")

def main():
    parser = argparse.ArgumentParser(description='Convert PKU-Market-PCB YOLO format to COCO format')
    parser.add_argument('--dataset_dir', type=str, required=True, 
                       help='Dataset directory containing train/val/test splits')
    parser.add_argument('--output_dir', type=str, default='./',
                       help='Output directory for JSON files')
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    
    # 转换训练集、验证集和测试集
    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split
        if split_dir.exists():
            output_file = output_dir / f'instances_{split}2017.json'
            print(f"\nConverting {split} set...")
            convert_dataset(str(split_dir), str(output_file), split)
        else:
            print(f"Warning: {split} directory not found")

if __name__ == "__main__":
    main()