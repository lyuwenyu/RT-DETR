FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git
RUN apt-get install -y curl
RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y

RUN pip3 install networkx==2.8.8
RUN pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
RUN pip3 uninstall onnxruntime -y
RUN pip3 install onnx==1.14.0 onnxruntime-gpu==1.15.1
RUN pip3 install pycocotools
# RUN pip3 install pycocotools==2.0.8
RUN pip3 install setuptools==59.6.0
RUN pip3 install scipy
# RUN pip3 install scipy==1.14.1
RUN pip3 install imgaug==0.4.0
RUN pip3 install PyYAML==6.0.2

RUN mkdir -p /app/repo/models
RUN curl -L -o /app/repo/models/rtdetr_r18vd_coco.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth
RUN curl -L -o /app/repo/models/rtdetr_r34vd_coco.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth
RUN curl -L -o /app/repo/models/rtdetr_r50vd_m_coco.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth
RUN curl -L -o /app/repo/models/rtdetr_r50vd_coco.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth
RUN curl -L -o /app/repo/models/rtdetr_r101vd_coco.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth
RUN curl -L -o /app/repo/models/rtdetr_r18vd_coco_objects365.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth
RUN curl -L -o /app/repo/models/rtdetr_r50vd_coco_objects365.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth
RUN curl -L -o /app/repo/models/rtdetr_r101vd_coco_objects365.pth https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_2x_coco_objects365_from_paddle.pth

RUN python3 -m pip install supervisely==6.73.216
RUN python3 -m pip install supervisely[model-benchmark]==6.73.216