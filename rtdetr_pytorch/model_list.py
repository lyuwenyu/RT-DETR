from copy import deepcopy


_models = [
    {
        "Model": "rtdetr_r18vd_coco",
        "dataset": "COCO",
        "AP_val": 46.4,
        "Params(M)": 20,
        "FPS(T4)": 217,
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"
        }
    },
    {
        "Model": "rtdetr_r34vd_coco",
        "dataset": "COCO",
        "AP_val": 48.9,
        "Params(M)": 31,
        "FPS(T4)": 161,
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth"
        }
    },
    {
        "Model": "rtdetr_r50vd_m_coco",
        "dataset": "COCO",
        "AP_val": 51.3,
        "Params(M)": 36,
        "FPS(T4)": 145,
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth"
        }
    },
    {
        "Model": "rtdetr_r50vd_coco",
        "dataset": "COCO",
        "AP_val": 53.1,
        "Params(M)": 42,
        "FPS(T4)": 108,
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth"
        }
    },
    {
        "Model": "rtdetr_r101vd_coco",
        "dataset": "COCO",
        "AP_val": 54.3,
        "Params(M)": 76,
        "FPS(T4)": 74,
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth"
        }
    },
    {
        "Model": "rtdetr_r18vd_coco_objects365",
        "dataset": "COCO+Objects365",
        "AP_val": 49.0,
        "Params(M)": 20,
        "FPS(T4)": 217,
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth"
        }
    },
    {
        "Model": "rtdetr_r50vd_coco_objects365",
        "dataset": "COCO+Objects365",
        "AP_val": 55.2,
        "Params(M)": 42,
        "FPS(T4)": 108,
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth"
        }
    },
    {
        "Model": "rtdetr_r101vd_coco_objects365",
        "dataset": "COCO+Objects365",
        "AP_val": 56.2,
        "Params(M)": 76,
        "FPS(T4)": 74,
        "meta": {
            "task_type": "object detection",
            "weights_url": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth"
        }
    }
]


def get_models():
    return deepcopy(_models)