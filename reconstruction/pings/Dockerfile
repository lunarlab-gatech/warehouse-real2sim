ARG PYTORCH="2.5.1"
ARG CUDA="11.8"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ARG DEBIAN_FRONTEND=noninteractive

# Set NVIDIA environment variables
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"

RUN apt-get install libglm-dev

WORKDIR /

COPY submodules /submodules
COPY requirements.txt .
RUN pip install -r requirements.txt

ARG USER_ID
ARG GROUP_ID

# Switch to same user as host system
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

WORKDIR /packages/pings
