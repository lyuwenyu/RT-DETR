def get_models():
    models = [
        {
            "name": "rtdetr_r18vd_coco",
            "dataset": "COCO",
            "FPS": 217,
            "meta": {
                "url": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"
            }
        }
    ]
    return models