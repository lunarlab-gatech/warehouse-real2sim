# Nvidia Neural Kernel Surface Reconstruction (NKSR) -- fat runtime image.
#
# Pattern: image bakes in EVERYTHING. The conda env, pytorch, pycg,
# torch-scatter, and the NKSR CUDA package (`pip install package/`,
# which compiles the CUDA extension and takes the bulk of the build
# time) are all installed at image-build time. The container is then
# started per-export as a fresh `--rm` run. Source code is NOT
# mounted -- NKSR is a stable library we consume, not a fork we patch.
# The NKSR source is cloned from GitHub at build time, pinned to a
# known-good commit, so this repo does not carry NVIDIA's source.
#
# Build once (rebuild only if the NKSR pin is bumped or driver flags change):
#   docker build -t nksr:dev .
#   docker build --build-arg NKSR_REF=<commit> -t nksr:dev .   # bump the pin
#
# Use it (per export):
#   docker run --gpus all --rm -it \
#     --user "$(id -u):$(id -g)" \
#     -v "$EXPORT_DIR":/work \
#     nksr:dev \
#     python /opt/nksr/run_nksr.py /work/nksr_config.yaml
#
# The driver script run_nksr.py lives at /opt/nksr/ in the image (COPY'd
# from this folder at build time). If you want to iterate on the driver
# without rebuilding the image, override it by mounting your edited copy:
#   -v "$PWD/run_nksr.py":/opt/nksr/run_nksr.py:ro

ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# Build toolchain + miniconda prerequisites + libGL bits pycg / open3d
# pulls in for headless mesh writing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        ca-certificates \
        bzip2 \
        libgl1 \
        libglib2.0-0 \
        libxrender1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Miniforge3 (not Miniconda) -- same `conda` binary, but ships with
# conda-forge as the only default channel. Sidesteps Anaconda's new
# Terms-of-Service gate on pkgs/main + pkgs/r that breaks non-interactive
# `conda env create` in Docker as of mid-2025. NKSR's environment.yml
# uses `nvidia/label/cuda-12.8.0` + `conda-forge` + `nodefaults`, so
# removing defaults entirely matches their intent.
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && conda config --set always_yes yes --set changeps1 no \
    && conda config --set channel_priority strict

# Clone NKSR from upstream at build time, pinned to a known-good commit
# (tip of nv-tlabs/nksr `public` as of Jul 2026: "use higher version of
# pycg"). The env + pip + package build all happen at image-build time;
# nothing from the host is needed. NKSR is distributed under the NVIDIA
# Source Code License (non-commercial research use) -- it lives in the
# image, not in this repository.
ARG NKSR_REF=14d75673b1506ffb39fc9c067ed581a0b0a3777d
RUN git clone https://github.com/nv-tlabs/nksr.git /opt/nksr_src \
    && git -C /opt/nksr_src checkout ${NKSR_REF}

# Create the conda env from NKSR's environment.yml + install python deps.
# The pip line picks up torch 2.7.0+cu128 and torch-scatter wheels from
# the indexes declared in requirements.txt.
RUN conda env create -f /opt/nksr_src/environment.yml \
    && conda run -n nksr pip install -r /opt/nksr_src/requirements.txt \
    && conda clean -afy

# Pycg ships visualization helpers (open3d, etc.) under its optional
# `[all]` extras. NKSR's requirements.txt only pulls in the bare
# `python-pycg`, but `from pycg import vis` runs at NKSR module import
# time (svh.py:17), so without open3d, `import nksr` itself fails. Pycg
# tells us exactly this in the error message it raises -- this line
# applies its prescribed fix once at build time.
RUN conda run -n nksr pip install -U 'python-pycg[all]'

# GPU architecture targets. Default = 8.6 (Ampere, RTX 30xx -- matches
# the host's RTX 3090 Ti and the -DCMAKE_CUDA_ARCHITECTURES=86 used for
# the GLIM build). Override at build time for other GPUs:
#   docker build --build-arg TORCH_CUDA_ARCH_LIST="8.9" -t nksr:dev .   # RTX 40xx
#   docker build --build-arg TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" ...     # multi-arch
#
# WHY this is needed in Docker: PyTorch's setup.py auto-detects target
# arches by reading the physically present GPU. `docker build` runs
# without --gpus all, so the detection returns an empty list and
# torch's _get_cuda_arch_flags crashes with IndexError. Setting the
# env var sidesteps the detection entirely.
ARG TORCH_CUDA_ARCH_LIST="8.6"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Pin CUDA_HOME to the base image's full toolkit (cublas, cusparse,
# cusolver, runtime libs, headers) instead of conda's nvcc-only package.
# Without this, torch picks /opt/conda/envs/nksr (just the compiler) and
# warns "No CUDA runtime is found" -- compile still works but linking
# pulls in conda's minimal libs instead of the full set.
ENV CUDA_HOME=/usr/local/cuda

# Build + install the NKSR CUDA package. --no-build-isolation lets the
# build see the conda env's nvcc + torch headers. This is the heaviest
# step (several minutes on first build).
RUN conda run -n nksr pip install --no-build-isolation /opt/nksr_src/package/

# Driver script lives at /opt/nksr/. Build context COPY happens after
# the heavy steps so iterating on the driver doesn't bust the cache.
RUN mkdir -p /opt/nksr
COPY run_nksr.py /opt/nksr/run_nksr.py

# Put the conda env's bin first on PATH so `python` and `nksr`'s deps
# resolve without a per-command `conda run -n nksr ...`.
ENV PATH=/opt/conda/envs/nksr/bin:${PATH}

# Default cwd is the user's mount point; the export folder mounts at /work.
WORKDIR /work

# Default action: print a usage line. The real invocation passes the
# YAML path explicitly.
CMD ["python", "/opt/nksr/run_nksr.py", "--help"]
