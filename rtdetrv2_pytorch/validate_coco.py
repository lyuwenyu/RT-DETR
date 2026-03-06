#!/usr/bin/env python3
"""
PKU-Market-PCB COCO Format Validator
验证COCO格式JSON文件的正确性
"""

import json
import os
from collections import defaultdict

def validate_coco_json(json_file, dataset_type="unknown"):
    """验证COCO格式JSON文件"""
    
    print(f"Validating {dataset_type} set: {json_file}")
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")
        return False
    
    # 检查必要字段
    required_fields = ['info', 'licenses', 'categories', 'images', 'annotations']
    for field in required_fields:
        if field not in data:
            print(f"❌ Missing required field: {field}")
            return False
    
    # 检查类别
    expected_categories = [
        'missing_hole', 'mouse_bite', 'open_circuit', 
        'short', 'spur', 'spurious_copper'
    ]
    
    if len(data['categories']) != len(expected_categories):
        print(f"❌ Expected {len(expected_categories)} categories, got {len(data['categories'])}")
        return False
    
    for i, category in enumerate(data['categories']):
        if category['name'] != expected_categories[i]:
            print(f"❌ Category mismatch at index {i}: expected {expected_categories[i]}, got {category['name']}")
            return False
        if category['id'] != i:
            print(f"❌ Category ID mismatch: expected {i}, got {category['id']}")
            return False
    
    # 检查图像
    image_ids = set()
    for img in data['images']:
        if 'id' not in img or 'file_name' not in img or 'width' not in img or 'height' not in img:
            print(f"❌ Missing required fields in image: {img}")
            return False
        
        if img['id'] in image_ids:
            print(f"❌ Duplicate image ID: {img['id']}")
            return False
        image_ids.add(img['id'])
        
        # 检查图像文件是否存在
        # 注意：这里不检查文件是否存在，因为路径可能不同
    
    # 检查标注
    annotation_ids = set()
    category_stats = defaultdict(int)
    
    for ann in data['annotations']:
        if 'id' not in ann or 'image_id' not in ann or 'category_id' not in ann or 'bbox' not in ann:
            print(f"❌ Missing required fields in annotation: {ann}")
            return False
        
        if ann['id'] in annotation_ids:
            print(f"❌ Duplicate annotation ID: {ann['id']}")
            return False
        annotation_ids.add(ann['id'])
        
        if ann['image_id'] not in image_ids:
            print(f"❌ Annotation refers to non-existent image_id: {ann['image_id']}")
            return False
        
        if ann['category_id'] >= len(expected_categories):
            print(f"❌ Invalid category_id: {ann['category_id']}")
            return False
        
        if len(ann['bbox']) != 4:
            print(f"❌ Invalid bbox format: {ann['bbox']}")
            return False
        
        # 检查bbox坐标
        x, y, w, h = ann['bbox']
        if w <= 0 or h <= 0:
            print(f"❌ Invalid bbox dimensions: width={w}, height={h}")
            return False
        
        category_stats[ann['category_id']] += 1
    
    # 统计信息
    print(f"✅ Validation passed!")
    print(f"   Images: {len(data['images'])}")
    print(f"   Annotations: {len(data['annotations'])}")
    print(f"   Categories: {len(data['categories'])}")
    
    print("\n   Category distribution:")
    for cat_id, count in sorted(category_stats.items()):
        print(f"     {expected_categories[cat_id]}: {count}")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate PKU-Market-PCB COCO format JSON files')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing COCO JSON files')
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    # 验证所有JSON文件
    json_files = [
        ('instances_train2017.json', 'train'),
        ('instances_val2017.json', 'val'),
        ('instances_test2017.json', 'test')
    ]
    
    all_valid = True
    
    for json_file, dataset_type in json_files:
        file_path = os.path.join(data_dir, json_file)
        if os.path.exists(file_path):
            if not validate_coco_json(file_path, dataset_type):
                all_valid = False
            print()
        else:
            print(f"⚠️  File not found: {file_path}")
            all_valid = False
    
    if all_valid:
        print("🎉 All JSON files are valid!")
    else:
        print("❌ Some JSON files have issues!")

if __name__ == "__main__":
    main()