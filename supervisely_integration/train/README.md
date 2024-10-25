<div align="center" markdown>

<img src=""/>  

# Train RT-DETR

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Obtain-saved-checkpoints">Obtain saved checkpoints</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/RT-DETR/supervisely_integration/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/RT-DETR)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/RT-DETR/supervisely_integration/train.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/RT-DETR/supervisely_integration/train.png)](https://supervise.ly)

</div>

# Overview

Train RT-DETR models in Supervisely on your custom data. All annotations will be converted to the bounding boxes automatically. Configure Train / Validation splits, model and training hyperparameters. Run on any agent (with GPU) in your team. Monitor progress, metrics, logs and other visualizations withing a single dashboard.

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

**Step 1.** Run the app from context menu of the project with annotations or from the Ecosystem

**Step 2.** Choose the pretrained or custom object detection model

<img src="https://github.com/user-attachments/assets/c236ced3-9165-4d5c-a2f0-2fb29edee05c" width="100%" style='padding-top: 10px'>  

**Step 3.** Select the classes you want to train RT-DETR on

<img src="https://github.com/user-attachments/assets/f0b2c84d-e2a8-4314-af4e-5ec7f784ce1f" width="100%" style='padding-top: 10px'>  

**Step 4.** Define the train/val splits

<img src="https://github.com/user-attachments/assets/3a2ac582-0489-493d-b2ff-8a98c94dfa20" width="100%" style='padding-top: 10px'>  

**Step 5.** Choose either ready-to-use augmentation template or provide custom pipeline

<img src="https://github.com/user-attachments/assets/a053fd89-4acc-44c0-af42-1ec0b84804a6" width="100%" style='padding-top: 10px'>  

**Step 6.** Configure the training parameters

<img src="https://github.com/user-attachments/assets/c5c715f0-836d-4613-a004-d139e2cf9706" width="100%" style='padding-top: 10px'>  

**Step 7.** Click `Train` button and observe the training progress, metrics charts and visualizations 

<img src="https://github.com/user-attachments/assets/703e182f-c84e-47de-8dc3-b01da8457580" width="100%" style='padding-top: 10px'>  

# Obtain saved checkpoints

All the trained checkpoints, that are generated through the process of training models are stored in [Team Files](https://app.supervise.ly/files/) in the folder **RT-DETR**.

You will see a folder thumbnail with a link to you saved checkpoints by the end of training process.

<img src="https://github.com/user-attachments/assets/6dd036f4-41de-4eb9-a87a-3387fb849ff1" width="100%" style='padding-top: 10px'>  

# Acknowledgment

This app is based on the great work `RT-DETR` ([github](https://github.com/lyuwenyu/RT-DETR)). ![GitHub Org's stars](https://img.shields.io/github/stars/lyuwenyu/RT-DETR?style=social)
