English | [简体中文](README_cn.md)


<h2 align="center">RT-DETR: DETRs Beat YOLOs on Real-time Object Detection</h2>
<p align="center">
    <a href="https://github.com/lyuwenyu/RT-DETR/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a>
    <a href="https://github.com/lyuwenyu/RT-DETR/pulls">
        <img alt="prs" src="https://img.shields.io/badge/PRs-Welcome-green">
    </a>
    <a href="https://github.com/lyuwenyu/RT-DETR/issues">
        <img alt="issues" src="https://img.shields.io/badge/Issues-2%20open-red">
    </a>
    <a href="https://arxiv.org/abs/2304.08069">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2304.08069-pink">
    </a>
    <a href="mailto: lyuwenyu@foxmail.com">
        <img alt="emal" src="https://img.shields.io/badge/Contact_me-Email-yellow">
    </a>
</p>

---

![ppdetr_overview](https://github.com/lyuwenyu/RT-DETR/assets/17582080/737f0d94-e028-4793-967e-201bdde57a5a)

This is the official implementation of the paper "[DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)".


## Updates!!!
- Release **RT-DETR-R50, RT-DETR-R101**
- Release **RT-DETR-R50-m（example for scaled)**
- Release **RT-DETR-R34, RT-DETR-R18**
- Release **RT-DETR-L, RT-DETR-X**


## Implementations
- [rtdetr-paddle](./rtdetr_paddle)
- [rtdetr-pytorch](./rtdetr_pytorch)


## Introduction
We propose a **R**eal-**T**ime **DE**tection **TR**ansformer (RT-DETR), the first real-time end-to-end object detector to our best knowledge. Our RT-DETR-L achieves 53.0% AP on COCO val2017 and 114 FPS on T4 GPU, while RT-DETR-X achieves 54.8% AP and 74 FPS, outperforming all YOLO detectors of the same scale in both speed and accuracy. Furthermore, our RT-DETR-R50 achieves 53.1% AP and 108 FPS, outperforming DINO-Deformable-DETR-R50 by 2.2% AP in accuracy and by about 21 times in FPS. 
<div align="center">
  <img src="https://github.com/PaddlePaddle/PaddleDetection/assets/17582080/3184a08e-aa4d-49cf-9079-f3695c4cc1c3" width=500 />
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
