from pathlib import Path

from src.core import YAMLConfig


def test_runtime_yml():
    config_file = Path(
        Path(__file__).parent.parent.parent.parent / "configs/runtime.yml"
    ).resolve()
    assert config_file.is_file()
    cfg = YAMLConfig(cfg_path=str(config_file))
    assert cfg
    assert cfg.yaml_cfg == {
        "ema": {"decay": 0.9999, "type": "ModelEMA", "warmups": 2000},
        "find_unused_parameters": False,
        "scaler": {"enabled": True, "type": "GradScaler"},
        "sync_bn": True,
        "use_amp": False,
        "use_ema": False,
    }
