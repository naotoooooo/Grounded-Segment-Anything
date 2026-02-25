# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
# FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-devel
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=1
ARG TORCH_ARCH=7.0

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda-11.1/

RUN mkdir -p /home/appuser/Grounded-Segment-Anything
COPY . /home/appuser/Grounded-Segment-Anything/

RUN rm -f /etc/apt/sources.list.d/cuda* \
    && rm -f /etc/apt/sources.list.d/nvidia*

# RUN apt-get update && apt-get install --no-install-recommends wget ffmpeg=7:* \
#     libsm6=2:* libxext6=2:* git=1:* nano=2.* \
#     vim=2:* -y \
#     && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        ffmpeg \
        libsm6 \
        libxext6 \
        git \
        nano \
        vim \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/appuser/Grounded-Segment-Anything

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev build-essential

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN python3 -m pip install --no-cache-dir -e segment_anything

# When using build isolation, PyTorch with newer CUDA is installed and can't compile GroundingDINO
RUN python3 -m pip install --no-cache-dir wheel maturin
RUN python3 -m pip install --no-cache-dir --no-build-isolation -e GroundingDINO
# RUN python3 -m pip install groundingdino-py

WORKDIR /home/appuser
# RUN pip install --no-cache-dir diffusers[torch]==0.15.1 opencv-python==4.7.0.72 \
#     pycocotools==2.0.6 matplotlib==3.5.3 \
#     onnxruntime==1.14.1 onnx==1.13.1 ipykernel==6.16.2 scipy gradio openai

RUN pip install --no-cache-dir diffusers[torch] opencv-python \
    pycocotools matplotlib \
    onnxruntime onnx ipykernel scipy gradio openai

RUN apt-get -y install x11-apps
RUN apt-get -y install mesa-utils