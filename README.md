English | [简体中文](README_cn.md)


<h2 align="center">RT-DETR: DETRs Beat YOLOs on Real-time Object Detection</h2>
<p align="center">
    <!-- <a href="https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a> -->
    <a href="https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/lyuwenyu/RT-DETR">
    </a>
    <a href="https://github.com/lyuwenyu/RT-DETR/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/lyuwenyu/RT-DETR">
    </a>
    <a href="https://github.com/lyuwenyu/RT-DETR/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/lyuwenyu/RT-DETR?color=pink">
    </a>
    <a href="https://github.com/lyuwenyu/RT-DETR">
        <img alt="issues" src="https://img.shields.io/github/stars/lyuwenyu/RT-DETR">
    </a>
    <a href="https://arxiv.org/abs/2304.08069">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2304.08069-red">
    </a>
    <a href="mailto: lyuwenyu@foxmail.com">
        <img alt="emal" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>
</p>

---
<details>
<summary>Fig</summary>

![ppdetr_overview](https://github.com/lyuwenyu/RT-DETR/assets/17582080/737f0d94-e028-4793-967e-201bdde57a5a)
</details>



This is the official implementation of the paper "[DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)".


## Updates!!!
- \[2023.11.07\] Add pytorch *rtdetr_r34vd* for requests [#107](https://github.com/lyuwenyu/RT-DETR/issues/107), [#114](https://github.com/lyuwenyu/RT-DETR/issues/114)
- \[2023.11.05\] upgrade the logic of `remap_mscoco_category` to facilitate training of custom datasets, see detils in [*Train custom data*](./rtdetr_pytorch/) part. [#81](https://github.com/lyuwenyu/RT-DETR/commit/95fc522fd7cf26c64ffd2ad0c622c392d29a9ebf)
- \[2023.10.23\] Add [*discussion for deployments*](https://github.com/lyuwenyu/RT-DETR/issues/95), supported onnxruntime, TensorRT, openVINO
- \[2023.10.12\] Add tuning code for pytorch version, now you can tuning rtdetr based on pretrained weights
- \[2023.09.19\] Upload [*pytorch weights*](https://github.com/lyuwenyu/RT-DETR/issues/42) convert from paddle version
- \[2023.08.24] Release rtdetr-18 pretrained models on objects365. *49.2 mAP* and *217 FPS*
- \[2023.08.22\] Upload *[rtdetr_pytorch](./rtdetr_pytorch/)* source code. Please enjoy it ❤️
- \[2023.08.15\] Release rtdetr-r101 pretrained models on objects365. *56.2 mAP* and *74 FPS*
- \[2023.07.30\] Release rtdetr-r50 pretrained models on objects365. *55.3 mAP* and *108 FPS*
- \[2023.07.28\] Fix some bugs, and add some comments. [1](https://github.com/lyuwenyu/RT-DETR/pull/14), [2](https://github.com/lyuwenyu/RT-DETR/commit/3b5cbcf8ae3b907e6b8bb65498a6be7c6736eabc)
- \[2023.07.13\] Upload [*training logs on coco*](https://github.com/lyuwenyu/RT-DETR/issues/8)
- \[2023.05.17\] Release RT-DETR-R18, RT-DETR-R34, RT-DETR-R50-m（example for scaled)
- \[2023.04.17\] Release RT-DETR-R50, RT-DETR-R101, RT-DETR-L, RT-DETR-X

## Implementations
- rtdetr paddle: [code](./rtdetr_paddle), [weights](./rtdetr_paddle)
- rtdetr pytorch: [code](./rtdetr_pytorch), [weights](./rtdetr_pytorch)


| Model | Epoch | Input shape | Dataset | $AP^{val}$ | $AP^{val}_{50}$| Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS)
|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|:---:|
| RT-DETR-R18 | 6x | 640 | COCO | 46.5 | 63.8 | 20 | 60 | 217 |
| RT-DETR-R34 | 6x | 640 | COCO | 48.9 | 66.8 | 31 | 92 | 161 |
| RT-DETR-R50-m | 6x | 640 | COCO | 51.3 | 69.6 | 36 | 100 | 145 |
| RT-DETR-R50 | 6x |  640 | COCO | 53.1 | 71.3 | 42 | 136 | 108 |
| RT-DETR-R101 | 6x | 640 | COCO | 54.3 | 72.7 | 76 | 259 | 74 |
| RT-DETR-HGNetv2-L | 6x | 640 | COCO | 53.0 | 71.6 | 32 | 110 | 114 |
| RT-DETR-HGNetv2-X | 6x | 640 | COCO | 54.8 | 73.1 | 67 | 234 | 74 |
| RT-DETR-R18 | 5x | 640 | COCO + Objects365 | **49.2** | **66.6** | 20 | 60 | **217** |
| RT-DETR-R50 | 2x | 640 | COCO + Objects365 | **55.3** | **73.4** | 42 | 136 | **108** |
| RT-DETR-R101 | 2x | 640 | COCO + Objects365 | **56.2** | **74.6** | 76 | 259 | **74** |

**Notes:**
- `COCO + Objects365` in the table means finetuned model on COCO using pretrained weights trained on Objects365.


## Introduction
We propose a **R**eal-**T**ime **DE**tection **TR**ansformer (RT-DETR, aka RTDETR), the first real-time end-to-end object detector to our best knowledge. Our RT-DETR-L achieves 53.0% AP on COCO val2017 and 114 FPS on T4 GPU, while RT-DETR-X achieves 54.8% AP and 74 FPS, outperforming all YOLO detectors of the same scale in both speed and accuracy. Furthermore, our RT-DETR-R50 achieves 53.1% AP and 108 FPS, outperforming DINO-Deformable-DETR-R50 by 2.2% AP in accuracy and by about 21 times in FPS. 
<div align="center">
  <!-- <img src="https://github.com/PaddlePaddle/PaddleDetection/assets/17582080/3184a08e-aa4d-49cf-9079-f3695c4cc1c3" width=300 /> -->
  <img src="https://github.com/lyuwenyu/RT-DETR/assets/17582080/42636690-1ecf-4647-b075-842ecb9bc562" width=500 >
</div>

## Citation
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
