English | [简体中文](README.md)

## Model Zoo

| Model | Epoch | backbone  | input shape | $AP^{val}$ | $AP^{val}_{50}$| Params(M) | FLOPs(G) |  T4 TensorRT FP16(FPS) | Pretrained Model | config |
|:--------------:|:-----:|:----------:| :-------:|:--------------------------:|:---------------------------:|:---------:|:--------:| :---------------------: |:------------------------------------------------------------------------------------:|:-------------------------------------------:|
| RT-DETR-R18 | 6x |  ResNet-18 | 640 | 46.5 | 63.8 | 20 | 60 | 217 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r18vd_dec3_6x_coco.pdparams) | [config](./rtdetr_r18vd_6x_coco.yml)
| RT-DETR-R34 | 6x |  ResNet-34 | 640 | 48.9 | 66.8 | 31 | 92 | 161 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r34vd_dec4_6x_coco.pdparams) | [config](./rtdetr_r34vd_6x_coco.yml)
| RT-DETR-R50-m | 6x |  ResNet-50 | 640 | 51.3 | 69.6 | 36 | 100 | 145 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_m_6x_coco.pdparams) | [config](./rtdetr_r50vd_m_6x_coco.yml)
| RT-DETR-R50 | 6x |  ResNet-50 | 640 | 53.1 | 71.3 | 42 | 136 | 108 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams) | [config](./rtdetr_r50vd_6x_coco.yml)
| RT-DETR-R101 | 6x |  ResNet-101 | 640 | 54.3 | 72.7 | 76 | 259 | 74 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_r101vd_6x_coco.pdparams) | [config](./rtdetr_r101vd_6x_coco.yml)
| RT-DETR-L | 6x |  HGNetv2 | 640 | 53.0 | 71.6 | 32 | 110 | 114 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_l_6x_coco.pdparams) | [config](rtdetr_hgnetv2_l_6x_coco.yml)
| RT-DETR-X | 6x |  HGNetv2 | 640 | 54.8 | 73.1 | 67 | 234 | 74 | [download](https://bj.bcebos.com/v1/paddledet/models/rtdetr_hgnetv2_x_6x_coco.pdparams) | [config](rtdetr_hgnetv2_x_6x_coco.yml)


**Notes:**
- RT-DETR uses 4 GPUs for training.
- RT-DETR was trained on COCO train2017 and evaluated on val2017.

## Quick start

<details open>
<summary>Install requirements</summary>

<!-- - PaddlePaddle == 2.4.2 -->
```bash
pip install -r requirements.txt
```

</details>

<details>
<summary>Data preparation</summary>

- Download and extract COCO 2017 train and val images.
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```
- Modify config [`dataset_dir`](configs/datasets/coco_detection.yml)
</details>


<details>
<summary>Training & Evaluation & Testing</summary>

- Training on a Single GPU:

```shell
# training on single-GPU
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --eval
```

- Training on Multiple GPUs:

```shell
# training on multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --fleet --eval
```

- Evaluation:

```shell
python tools/eval.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams
```

- Inference:

```shell
python tools/infer.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams \
              --infer_img=./demo/000000570688.jpg
```

</details>

## Deploy

<details open>
<summary>1. Export model </summary>

```shell
cd PaddleDetection
python tools/export_model.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
              -o weights=https://bj.bcebos.com/v1/paddledet/models/rtdetr_r50vd_6x_coco.pdparams trt=True \
              --output_dir=output_inference
```

</details>

<details>
<summary>2. Convert to ONNX </summary>

- Install [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX) and ONNX

```shell
pip install onnx==1.13.0
pip install paddle2onnx==1.0.5
```

- Convert:

```shell
paddle2onnx --model_dir=./output_inference/rtdetr_r50vd_6x_coco/ \
            --model_filename model.pdmodel  \
            --params_filename model.pdiparams \
            --opset_version 16 \
            --save_file rtdetr_r50vd_6x_coco.onnx
```
</details>

<details>
<summary>3. Convert to TensorRT </summary>

- TensorRT version >= 8.5.1
- Inference can refer to [Bennchmark](../benchmark)

```shell
trtexec --onnx=./rtdetr_r50vd_6x_coco.onnx \
        --workspace=4096 \
        --shapes=image:1x3x640x640 \
        --saveEngine=rtdetr_r50vd_6x_coco.trt \
        --avgRuns=100 \
        --fp16
```

-
</details>


## Others

<details>
<summary>1. Parameters and FLOPs </summary>

1. Find and modify [paddle flops source code](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/hapi/dynamic_flops.py#L28)

```python
# anaconda3/lib/python3.8/site-packages/paddle/hapi/dynamic_flops.py
def flops(net, input_size, inputs=None, custom_ops=None, print_detail=False):
    if isinstance(net, nn.Layer):
        # If net is a dy2stat model, net.forward is StaticFunction instance,
        # we set net.forward to original forward function.
        _, net.forward = unwrap_decorators(net.forward)

        # by lyuwenyu
        if inputs is None:
            inputs = paddle.randn(input_size)

        return dynamic_flops(
            net, inputs=inputs, custom_ops=custom_ops, print_detail=print_detail
        )
    elif isinstance(net, paddle.static.Program):
        return static_flops(net, print_detail=print_detail)
    else:
        warnings.warn(
            "Your model must be an instance of paddle.nn.Layer or paddle.static.Program."
        )
        return -1
```

2. Run below code

```python
import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.core.workspace import create

cfg_path = './configs/rtdetr/rtdetr_r50vd_6x_coco.yml'
cfg = load_config(cfg_path)
model = create(cfg.architecture)

blob = {
    'image': paddle.randn([1, 3, 640, 640]),
    'im_shape': paddle.to_tensor([[640, 640]]),
    'scale_factor': paddle.to_tensor([[1., 1.]])
}
paddle.flops(model, None, blob, custom_ops=None, print_detail=False)

# Outpus
# Total Flops: 68348108800     Total Params: 41514204

```


</details>


## Citing RT-DETR
If you use `RT-DETR` in your work, please use the following BibTeX entries:
```
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Wenyu Lv and Shangliang Xu and Yian Zhao and Guanzhong Wang and Jinman Wei and Cheng Cui and Yuning Du and Qingqing Dang and Yi Liu},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
