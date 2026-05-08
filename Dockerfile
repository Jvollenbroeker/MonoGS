# =============================================================================
# MonoGS — Docker image for NVIDIA L4 (Ada Lovelace, sm_89)
#
# Why this stack:
#   * The repo's environment.yml targets CUDA 11.6 / PyTorch 1.12 / Python 3.7.
#     CUDA 11.6 has no native sm_89 kernels, so on an L4 the CUDA submodules
#     either fail to build with the right arch or fall back to slow PTX JIT.
#   * Minimum CUDA for native sm_89 support is 11.8.
#   * PyTorch 2.0+ (which is what ships against cu118) requires Python >= 3.8,
#     so Python is bumped to 3.10. Open3D 0.17.0, plyfile, evo, etc. all
#     still support 3.10, so the rest of environment.yml carries over.
#   * Host driver 580.x supports any CUDA <= 13, so a cu11.8 container is
#     fully compatible with your VM.
# =============================================================================

ARG CUDA_VERSION=11.8.0
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# -----------------------------------------------------------------------------
# System dependencies
#   build-essential / cmake / ninja : compiling the CUDA submodules
#   libgl*, libegl*, libxi/xrandr.. : Open3D + PyOpenGL + glfw GUI runtime
#   libglib2.0-0, libsm6, libxext6  : OpenCV runtime
#   libusb-1.0-0                    : pyrealsense2 (optional live demo)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        git \
        wget \
        curl \
        ca-certificates \
        unzip \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libglu1-mesa \
        libosmesa6 \
        libegl1 \
        libgles2 \
        libxi6 \
        libxinerama1 \
        libxcursor1 \
        libxrandr2 \
        libxxf86vm1 \
        libx11-6 \
        libxkbcommon0 \
        libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Miniconda (Python 3.10)
# -----------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN wget -qO /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda config --system --set always_yes true \
    && conda config --system --set channel_priority flexible \
    && conda clean -afy

# -----------------------------------------------------------------------------
# Conda env "MonoGS"
# -----------------------------------------------------------------------------
ENV CONDA_ENV=MonoGS
RUN conda create -n ${CONDA_ENV} python=3.10 pip=23.3 \
    && conda clean -afy

# Use the env for every following RUN.
SHELL ["conda", "run", "--no-capture-output", "-n", "MonoGS", "/bin/bash", "-c"]

# -----------------------------------------------------------------------------
# PyTorch 2.0.1 + CUDA 11.8 (matches the cuda:11.8.0-devel base, sm_89-capable)
# -----------------------------------------------------------------------------
#
# setuptools is pinned to <70: PyTorch 2.0.1's torch/utils/cpp_extension.py
# imports `from pkg_resources import packaging`, which setuptools 70+
# removed. Without this pin the submodule build fails with
# "ModuleNotFoundError: No module named 'pkg_resources'".
RUN pip install --upgrade "pip<24" "setuptools==69.5.1" "wheel" \
    && pip install \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        torchaudio==2.0.2+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# -----------------------------------------------------------------------------
# Python deps from environment.yml (pip section + the few conda packages
# that are pure-python on PyPI). cudatoolkit is *not* needed: PyTorch ships
# its own runtime libs and the system CUDA toolkit (from the base image)
# provides nvcc for the submodules below.
#
# numpy is pinned to <2: PyTorch 2.0.1, OpenCV 4.8.1, Open3D 0.17 and the
# CUDA submodules below are all compiled against NumPy 1.x ABI. NumPy 2.0
# is an ABI break — without this pin you get
# "AttributeError: _ARRAY_API not found" / "numpy.core.multiarray failed
# to import" on every native module that touches arrays.
#
# matplotlib is pinned to 3.6.x: evo 1.11.0's traj_colormap calls
# fig.colorbar(mappable) without ax=. matplotlib 3.7 made that a hard
# error ("Unable to determine Axes to steal space for Colorbar"); 3.6.x
# still does the implicit plt.gca() lookup that evo relies on.
# -----------------------------------------------------------------------------
RUN pip install "numpy==1.26.4" "matplotlib==3.6.3" \
    && pip install \
        plyfile==0.8.1 \
        tqdm \
        opencv-python==4.8.1.78 \
        munch \
        trimesh \
        evo==1.11.0 \
        open3d==0.17.0 \
        torchmetrics \
        imgviz \
        PyOpenGL \
        glfw \
        PyGLM \
        wandb \
        lpips \
        rich \
        ruff \
    && python -c "import numpy, matplotlib; assert numpy.__version__.startswith('1.'), numpy.__version__; assert matplotlib.__version__.startswith('3.6.'), matplotlib.__version__; print('numpy', numpy.__version__, 'matplotlib', matplotlib.__version__, 'OK')"

# -----------------------------------------------------------------------------
# Build the CUDA submodules with the L4 architecture baked in.
# TORCH_CUDA_ARCH_LIST="8.9" => Ada Lovelace (L4, RTX 40-series).
# Add other archs space-separated if you also build for other GPUs.
# -----------------------------------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST="8.9" \
    FORCE_CUDA=1 \
    MPLBACKEND=Agg

# evo (trajectory eval) calls `matplotlib.use(SETTINGS.plot_backend)` at
# import time, which overrides MPLBACKEND. Its default plot_backend is
# TkAgg, which fails inside a headless container. Bake plot_backend=Agg
# into evo's user settings during the build so the eval subprocess can
# render the trajectory plot to PNG instead of crashing.
RUN evo_config set plot_backend Agg \
    && evo_config show | grep plot_backend

WORKDIR /workspace/MonoGS

# Copy only the submodules first so a code edit doesn't invalidate the
# (slow) CUDA-extension build cache.
#
# --no-build-isolation: both submodules' setup.py do `import torch` at
# module level, but PEP 517 build isolation creates a fresh venv with no
# torch installed -> ModuleNotFoundError. Disabling isolation makes the
# build use the env's already-installed torch.
COPY submodules /workspace/MonoGS/submodules
RUN pip install --no-build-isolation ./submodules/simple-knn \
    && pip install --no-build-isolation ./submodules/diff-gaussian-rasterization

# Now copy the rest of the project.
COPY . /workspace/MonoGS

# -----------------------------------------------------------------------------
# Make `MonoGS` the default env for interactive shells and `docker run` CMDs.
# -----------------------------------------------------------------------------
ENV PATH=${CONDA_DIR}/envs/${CONDA_ENV}/bin:${PATH} \
    CONDA_DEFAULT_ENV=${CONDA_ENV}

RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}" \
        >> /etc/bash.bashrc

SHELL ["/bin/bash", "-lc"]
CMD ["bash"]
