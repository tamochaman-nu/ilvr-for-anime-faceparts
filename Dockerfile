# ホストのドライバに合わせて切り替える:
#   >= 550: cuda12.4 / >= 530: cuda12.1 / >= 520: cuda11.8
ARG BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# mpi4py のビルドに OpenMPI が要る (guided_diffusion の dist_util / image_datasets が import する)
RUN apt-get update && apt-get install -y \
    openmpi-bin libopenmpi-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# /app はバインドマウントされるので pip install -e せず PYTHONPATH で読み込む
ENV PYTHONPATH=/app

# torch / torchvision はベースイメージのものを使う
# lpips / scikit-image / torch-fidelity は Phase 3 の評価 (LPIPS, Lab 色差, KID) 用
RUN pip install --no-cache-dir \
    "blobfile>=1.0.5" tqdm mpi4py Pillow \
    lpips==0.1.4 scikit-image==0.24.0 torch-fidelity==0.3.0

CMD ["/bin/bash"]
