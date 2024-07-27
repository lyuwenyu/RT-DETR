# DETRs Beat YOLOs on Real-time Object Detection

## Introduction
This repository is the official pytorch implementation of [*RTDETR*](https://arxiv.org/abs/2304.08069v1), and is compatiable with [RT-DETR/rtdetr_pytorch](https://github.com/lyuwenyu/RT-DETR/tree/main). For paddle version implementation, please refer to [RT-DETR/rtdetr_paddle](https://github.com/lyuwenyu/RT-DETR/tree/main). **If you are using rtdetr for the first time, it is highly recommended to use [rtdetrv2](../rtdetrv2/)**.

<details open>
<summary> Fig </summary>
<div align="center">
  <img src="https://github.com/lyuwenyu/RT-DETR/assets/17582080/42636690-1ecf-4647-b075-842ecb9bc562" width=500>
</div>
</details>

<!-- 
<div align="center">
  <img src="https://github.com/lyuwenyu/RT-DETR/assets/17582080/42636690-1ecf-4647-b075-842ecb9bc562" width=500>
</div> -->


## Model Zoo
| Model | Dataset | Input Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | #Params(M) | FPS |  checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
rtdetr_r18vd | COCO | 640 | 46.4 | 63.7 | 20 | 217 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth)
rtdetr_r34vd | COCO | 640 | 48.9 | 66.8 | 31 | 161 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth)
rtdetr_r50vd_m | COCO | 640 | 51.3 | 69.5 | 36 | 145 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth)
rtdetr_r50vd | COCO | 640 | 53.1 | 71.2| 42 | 108 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth)
rtdetr_r101vd | COCO | 640 | 54.3 | 72.8 | 76 | 74 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth)
rtdetr_18vd | COCO+Objects365 | 640 | 49.0 | 66.5 | 20 | 217 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth)
rtdetr_r50vd | COCO+Objects365 | 640 | 55.2 | 73.4 | 42 | 108 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth)
rtdetr_r101vd | COCO+Objects365 | 640 | 56.2 | 74.5 | 76 | 74 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth)

<!-- rtdetr_r18vd | COCO | 640 | 46.5 | 63.6 | 20 | 217 | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_6x_coco.pth) -->

<!-- rtdetr_r18vd | Objects365 | 640 | 22.9 |  31.2| - | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth)
rtdetr_r50vd | Objects365 | 640 | 35.1 | 46.2 | - | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth)
rtdetr_r101vd | Objects365 | 640 | 36.8 | 48.3 | - | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth) -->

Notes
<!-- - AP is evaluated on coco 2017 val dataset -->
<!-- RT-DETR was trained on COCO train2017 and evaluated on val2017. -->
- `COCO + Objects365` in the table means finetuned model on `COCO` using pretrained weights trained on `Objects365`.
- `FPS` is evaluated on a single T4 GPU with $batch\\_size = 1$ and $tensorrt\\_fp16$ mode
- `url`<sup>`*`</sup> is the url of the pretrained weights, converted from the paddle model to save energy. *There may be slight differences between this table and the paper.


## Usage
<details>
<summary> details </summary>

<!-- <summary>1. Training </summary> -->
1. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config &> log.txt 2>&1 &
```

<!-- <summary>2. Testing </summary> -->
2. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config -r path/to/checkpoint --test-only
```

<!-- <summary>3. Tuning </summary> -->
3. Tuning
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config -t path/to/checkpoint &> log.txt 2>&1 &
```

<!-- <summary>4. Export onnx </summary> -->
4. Export onnx
```shell
python tools/export_onnx.py -c path/to/config -r path/to/checkpoint --check
```

<!-- <summary>5. Inference </summary> -->
5. Inference

Support torch, onnxruntime, tensorrt and openvino, see details in *references/deploy*
```shell
python references/deploy/rtdetrv2_onnx.py --onnx-file=model.onnx --im-file=xxxx
python references/deploy/rtdetrv2_tensorrt.py --trt-file=model.trt --im-file=xxxx
python references/deploy/rtdetrv2_torch.py -c path/to/config -r path/to/checkpoint --im-file=xxx --device=cuda:0
```
</details>


## Citation
If you use `RTDETR` in your work, please use the following BibTeX entries:

<details>
<summary> bibtex </summary>

```latex
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@software{Lv_rtdetr_by_cvperception_2023,
author = {Lv, Wenyu},
license = {Apache-2.0},
month = oct,
title = {{rtdetr by cvperception}},
url = {https://github.com/lyuwenyu/cvperception/},
version = {0.0.1dev},
year = {2023}
}
```
</details>
