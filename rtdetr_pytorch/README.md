## TODO
<details>
<summary> see details </summary>

- [x] Training
- [x] Evaluation
- [x] Export onnx
- [x] Upload source code
- [x] Upload weight convert from paddle, see [*links*](https://github.com/lyuwenyu/RT-DETR/issues/42)
- [x] Align training details with the [*paddle version*](../rtdetr_paddle/)
- [x] Tuning rtdetr based on [*pretrained weights*](https://github.com/lyuwenyu/RT-DETR/issues/42)

</details>


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
rtdetr_regnet | COCO | 640 | 51.6 | 69.6 | 38 | 67 | [url<sup>*</sup>](https://drive.google.com/file/d/1K2EXJgnaEUJcZCLULHrZ492EF4PdgVp9/view?usp=sharing)
rtdetr_dla34 | COCO | 640 | 49.6 | 67.4  | 34 | 83 | [url<sup>*</sup>](https://drive.google.com/file/d/1_rVpl-jIelwy2LDT3E4vdM4KCLBcOtzZ/view?usp=sharing)

Notes
- `COCO + Objects365` in the table means finetuned model on `COCO` using pretrained weights trained on `Objects365`.
- `url`<sup>`*`</sup> is the url of pretrained weights convert from paddle model for save energy. *It may have slight differences between this table and paper*
<!-- - `FPS` is evaluated on a single T4 GPU with $batch\\_size = 1$ and $tensorrt\\_fp16$ mode -->

## Quick start

<details>
<summary>Install</summary>

```bash
pip install -r requirements.txt
```

</details>


<details>
<summary>Data</summary>

- Download and extract COCO 2017 train and val images.
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```
- Modify config [`img_folder`, `ann_file`](configs/dataset/coco_detection.yml)
</details>



<details>
<summary>Training & Evaluation</summary>

- Training on a Single GPU:

```shell
# training on single-gpu
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

- Training on Multiple GPUs:

```shell
# train on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

- Evaluation on Multiple GPUs:

```shell
# val on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint --test-only
```

</details>



<details>
<summary>Export</summary>

```shell
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -r path/to/checkpoint --check
```
</details>




<details open>
<summary>Train custom data</summary>

1. set `remap_mscoco_category: False`. This variable only works for ms-coco dataset. If you want to use `remap_mscoco_category` logic on your dataset, please modify variable [`mscoco_category2name`](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/data/coco/coco_dataset.py#L154) based on your dataset.

2. add `-t path/to/checkpoint` (optinal) to tuning rtdetr based on pretrained checkpoint. see [training script details](./tools/README.md).
</details>
