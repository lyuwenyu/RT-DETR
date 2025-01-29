FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS base

FROM base AS train

RUN apt-get update && apt-get install -y git

WORKDIR /app

COPY rtdetrv2_pytorch/requirements.txt requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt
