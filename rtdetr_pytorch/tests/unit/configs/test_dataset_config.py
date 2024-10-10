from pathlib import Path

from src.core import YAMLConfig


def test_coco_detection_yml():
    coco_detection_config_file = Path(
        Path(__file__).parent.parent.parent.parent
        / "configs/dataset/coco_detection.yml"
    ).resolve()
    assert coco_detection_config_file.is_file()
    cfg = YAMLConfig(cfg_path=str(coco_detection_config_file))
    assert cfg
    assert cfg.yaml_cfg == {
        "task": "detection",
        "num_classes": 80,
        "remap_mscoco_category": True,
        "train_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": "./dataset/coco/train2017/",
                "ann_file": "./dataset/coco/annotations/instances_train2017.json",
                "transforms": {"type": "Compose", "ops": None},
            },
            "shuffle": True,
            "batch_size": 8,
            "num_workers": 4,
            "drop_last": True,
        },
        "val_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": "./dataset/coco/val2017/",
                "ann_file": "./dataset/coco/annotations/instances_val2017.json",
                "transforms": {"type": "Compose", "ops": None},
            },
            "shuffle": False,
            "batch_size": 8,
            "num_workers": 4,
            "drop_last": False,
        },
    }
