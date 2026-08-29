FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST="8.9" \
    CUDA_HOME=/usr/local/cuda PYTHONUNBUFFERED=1 MAX_JOBS=4 \
    PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
      git wget curl unzip ca-certificates build-essential cmake ninja-build pkg-config \
      python3 python3-pip python3-dev python-is-python3 \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libegl1 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
# Keep numpy on the 1.x ABI for the compiled CUDA extensions below
RUN pip install "numpy<2"

# basic (mirrors setup.sh --basic)
RUN pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja \
      rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph && \
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# attention + sparse conv
RUN pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install packaging ninja einops psutil
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.4cxx11abiFalse-cp310-cp310-linux_x86_64.whl
RUN pip install spconv-cu118

# rasterizers (compiled for SM 8.9 at build via TORCH_CUDA_ARCH_LIST; need torch visible → --no-build-isolation)
RUN git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && \
    pip install --no-build-isolation /tmp/nvdiffrast
RUN git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/doct && \
    pip install --no-build-isolation /tmp/doct
RUN git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip && \
    pip install --no-build-isolation /tmp/mip/submodules/diff-gaussian-rasterization/

# kaolin (repo ships a cu121 wheel for torch 2.4.0)
RUN pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# Qwen2.5-VL deps LAST (pins transformers 4.50.0)
RUN pip install transformers==4.50.0 qwen-vl-utils 'accelerate>=0.26.0'

# Pure-Python deps the repo's inference scripts import (not in setup.sh): debugger + decoder-path libs
RUN pip install ipdb pandas plyfile lpips

# Fix: nvdiffrast 0.4.0 needs setuptools>=64 or it builds as package "UNKNOWN"
# and `import nvdiffrast` fails. Upgrade setuptools, then reinstall cleanly.
RUN pip uninstall -y UNKNOWN nvdiffrast >/dev/null 2>&1; \
    pip install --upgrade setuptools wheel && \
    git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast2 && \
    pip install --no-build-isolation /tmp/nvdiffrast2

WORKDIR /workspace
CMD ["/bin/bash"]
