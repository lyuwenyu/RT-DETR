
## Quick start

<details >
<summary>Setup</summary>

```shell

pip install -r requirements.txt
```

The following is the corresponding `torch` and `torchvision` versions.
`rtdetr` | `torch` | `torchvision`
|---|---|---|
| `-` | `2.4` | `0.19` |
| `-` | `2.2` | `0.17` |
| `-` | `2.1` | `0.16` |
| `-` | `2.0` | `0.15` |

</details>

<details open>
<summary>Fig</summary>

<div align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/437877e9-1d4f-4d30-85e8-aafacfa0ec56">
</div>

</details>


## Model Zoo

### Base models

| Model | Dataset | Input Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | #Params(M) | FPS | config| checkpoint | 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
**RT-DETRv2-S** | COCO | 640 | **48.1** <font color=green>(+1.6)</font> | **65.1** | 20 | 217 | [config](./configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth) |
**RT-DETRv2-M**<sup>*<sup> | COCO | 640 | **49.9** <font color=green>(+1.0)</font> | **67.5** | 31 | 161 | [config](./configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth)
**RT-DETRv2-M** | COCO | 640 | **51.9** <font color=green>(+0.6)</font> | **69.9** | 36 | 145 | [config](./configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth)
**RT-DETRv2-L** | COCO | 640 | **53.4** <font color=green>(+0.3)</font> | **71.6** | 42 | 108 | [config](./configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth)
**RT-DETRv2-X** | COCO | 640 | 54.3 | **72.8** <font color=green>(+0.1)</font> | 76 | 74 | [config](./configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth)
<!-- rtdetrv2_hgnetv2_l | COCO | 640 | 52.9 | 71.5 | 32 | 114 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_l_6x_coco_from_paddle.pth) 
rtdetrv2_hgnetv2_x | COCO | 640 | 54.7 | 72.9 | 67 | 74 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_x_6x_coco_from_paddle.pth) 
rtdetrv2_hgnetv2_h | COCO | 640 | 56.3 | 74.8 | 123 | 40 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_hgnetv2_h_6x_coco_from_paddle.pth) 
rtdetrv2_18vd | COCO+Objects365 | 640 | 49.0 | 66.5 | 20 | 217 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_5x_coco_objects365_from_paddle.pth)
rtdetrv2_r50vd | COCO+Objects365 | 640 | 55.2 | 73.4 | 42 | 108 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_2x_coco_objects365_from_paddle.pth)
rtdetrv2_r101vd | COCO+Objects365 | 640 | 56.2 | 74.5 | 76 | 74 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_2x_coco_objects365_from_paddle.pth)
 -->

**Notes:**
- `AP` is evaluated on *MSCOCO val2017* dataset.
- `FPS` is evaluated on a single T4 GPU with $batch\\_size = 1$, $fp16$, and $TensorRT>=8.5.1$.
- `COCO + Objects365` in the table means finetuned model on `COCO` using pretrained weights trained on `Objects365`.



### Models of discrete sampling

| Model | Sampling Method | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | config| checkpoint 
| :---: | :---: | :---: | :---: | :---: | :---: |
**RT-DETRv2-S_dsp** | discrete_sampling | 47.4 | 64.8 <font color=red>(-0.1)</font> | [config](./configs/rtdetrv2/rtdetrv2_r18vd_dsp_3x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_dsp_3x_coco.pth)
**RT-DETRv2-M**<sup>*</sup>**_dsp** | discrete_sampling | 49.2 | 67.1 <font color=red>(-0.4)</font> | [config](./configs/rtdetrv2/rtdetrv2_r34vd_dsp_1x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rrtdetrv2_r34vd_dsp_1x_coco.pth)
**RT-DETRv2-M_dsp** | discrete_sampling | 51.4 | 69.7 <font color=red>(-0.2)</font> | [config](./configs/rtdetrv2/rtdetrv2_r50vd_m_dsp_3x_coco.yml) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_dsp_3x_coco.pth)
**RT-DETRv2-L_dsp** | discrete_sampling | 52.9 | 71.3 <font color=red>(-0.3)</font> |[config](./configs/rtdetrv2/rtdetrv2_r50vd_dsp_1x_coco.yml)| [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_dsp_1x_coco.pth)


<!-- **rtdetrv2_r18vd_dsp1** | discrete_sampling | 21600 | 46.3 | 63.9 | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_dsp1_1x_coco.pth) -->

<!-- rtdetrv2_r18vd_dsp1 | discrete_sampling | 21600 | 45.5 | 63.0 | 4.34 | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_dsp1_120e_coco.pth) -->
<!-- 4.3 -->

**Notes:**
- The impact on inference speed is related to specific device and software.
- `*_dsp*` is the model inherit `*_sp*` model's knowledge and adapt to `discrete_sampling` strategy. **You can use TensorRT 8.4 (or even older versions) to inference for these models**
<!-- - `grid_sampling` use `grid_sample` to sample attention map, `discrete_sampling` use `index_select` method to sample attention map.  -->


### Ablation on sampling points

<!-- Flexible samping strategy in cross attenstion layer for devices that do **not** optimize (or not support) `grid_sampling` well. You can choose models based on specific scenarios and the trade-off between speed and accuracy. -->

| Model | Sampling Method | #Points | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | checkpoint 
| :---: | :---: | :---: | :---: | :---: | :---: |
**rtdetrv2_r18vd_sp1** | grid_sampling | 21,600 | 47.3 | 64.3 <font color=red>(-0.6) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp1_120e_coco.pth)
**rtdetrv2_r18vd_sp2** | grid_sampling | 43,200 | 47.7 | 64.7 <font color=red>(-0.2) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp2_120e_coco.pth)
**rtdetrv2_r18vd_sp3** | grid_sampling | 64,800 | 47.8 | 64.8 <font color=red>(-0.1) | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_sp3_120e_coco.pth)
rtdetrv2_r18vd(_sp4)| grid_sampling | 86,400 | 47.9 | 64.9 | [url](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r18vd_120e_coco.pth) 

**Notes:**
- The impact on inference speed is related to specific device and software.
- `#points` the total number of sampling points in decoder for per image inference.


## Usage
<details>
<summary> details </summary>

<!-- <summary>1. Training </summary> -->
1. Training
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config --use-amp --seed=0 &> log.txt 2>&1 &
```

<!-- <summary>2. Testing </summary> -->
2. Testing
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config -r path/to/checkpoint --test-only
```

<!-- <summary>3. Tuning </summary> -->
3. Tuning
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=9909 --nproc_per_node=4 tools/train.py -c path/to/config -t path/to/checkpoint --use-amp --seed=0 &> log.txt 2>&1 &
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
If you use `RTDETR` or `RTDETRv2` in your work, please use the following BibTeX entries:

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

@misc{lv2024rtdetrv2improvedbaselinebagoffreebies,
      title={RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer}, 
      author={Wenyu Lv and Yian Zhao and Qinyao Chang and Kui Huang and Guanzhong Wang and Yi Liu},
      year={2024},
      eprint={2407.17140},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17140}, 
}
```
</details>
