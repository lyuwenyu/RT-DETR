<div align="center" markdown>

<img src="https://github.com/user-attachments/assets/0d12aa66-a97e-485b-aa16-a70ca3c924fa"/>  

# Serve RT-DETR

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Obtain-saved-checkpoints">Obtain saved checkpoints</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/RT-DETR/supervisely_integration/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/RT-DETR)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/RT-DETR/supervisely_integration/serve.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/RT-DETR/supervisely_integration/serve.png)](https://supervise.ly)

</div>

# Overview

Serve RT-DETR model as a Supervisely Application. RT-DETR is a real-time object detection model that combines the advantages of DETR and YOLO. It is based on the idea of DETR, which uses transformer to directly predict object queries, and the idea of YOLO, which uses anchor boxes to predict object locations. RT-DETR is able to achieve real-time inference speed and high accuracy on object detection tasks.

Learn more about RT-DETR and available models [here](https://github.com/lyuwenyu/RT-DETR).

# Model Zoo

|     Model      |     Dataset     | Input Size | AP<sup>val</sup> | AP<sub>50</sub><sup>val</sup> | #Params(M) | FPS |                                                           checkpoint                                                           |
|:--------------:|:---------------:|:----------:|:----------------:|:-----------------------------:|:----------:|:---:|:------------------------------------------------------------------------------------------------------------------------------:|
|  rtdetr_r18vd  |      COCO       |    640     |       46.4       |             63.7              |     20     | 217 |    [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth)     |
|  rtdetr_r34vd  |      COCO       |    640     |       48.9       |             66.8              |     31     | 161 |    [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth)     |
| rtdetr_r50vd_m |      COCO       |    640     |       51.3       |             69.5              |     36     | 145 |      [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth)      |
|  rtdetr_r50vd  |      COCO       |    640     |       53.1       |             71.2              |     42     | 108 |       [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth)       |
| rtdetr_r101vd  |      COCO       |    640     |       54.3       |             72.8              |     76     | 74  |      [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth)       |
|  rtdetr_18vd   | COCO+Objects365 |    640     |       49.0       |             66.5              |     20     | 217 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth)  |
|  rtdetr_r50vd  | COCO+Objects365 |    640     |       55.2       |             73.4              |     42     | 108 | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth)  |
| rtdetr_r101vd  | COCO+Objects365 |    640     |       56.2       |             74.5              |     76     | 74  | [url<sup>*</sup>](https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth) |

Notes
- `COCO + Objects365` in the table means finetuned model on `COCO` using pretrained weights trained on `Objects365`.
- `url`<sup>`*`</sup> is the url of pretrained weights convert from paddle model for save energy. *It may have slight differences between this table and paper*

# How to Run

0. Start the application from an app's context menu or the Ecosystem.

1. Select pre-trained model or custom model trained inside Supervisely platform, and a runtime for inference.

<img src="https://github.com/user-attachments/assets/cb862358-4c1f-4357-8d97-21bb96dca1f7" />

2. Select device and press the `Serve` button, then wait for the model to deploy.

<img src="https://github.com/user-attachments/assets/3259f249-1328-4c96-bc18-c3d9155b8512" />

3. You will see a message once the model has been successfully deployed.

<img src="https://github.com/user-attachments/assets/37a0fc45-4b9e-48b5-9b5e-1178c2169556" />

# Acknowledgment

This app is based on the great work `RT-DETR` ([github](https://github.com/lyuwenyu/RT-DETR)). ![GitHub Org's stars](https://img.shields.io/github/stars/lyuwenyu/RT-DETR?style=social)
