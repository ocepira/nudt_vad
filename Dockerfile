
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

WORKDIR /project
ENTRYPOINT []
# 设置时区
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone

# ✅ NVIDIA官方正版环境变量 - 彻底禁用所有CUDA相关日志/版权声明 (精简为1份，无重复)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_DISABLE_RECOMMENDATIONS=1
SHELL [ "/bin/bash", "--login", "-c" ]


# 修复NVIDIA GPG密钥
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 安装系统依赖
RUN echo "deb http://mirrors.aliyun.com/ubuntu/ focal main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update -q && \
    apt-get install -q -y --no-install-recommends \
    wget curl net-tools python3-dev python3-pip python3-tk python3.8-venv \
    libreadline-dev libncurses5-dev libncursesw5-dev libunistring-dev git \
    build-essential gcc g++ ninja-build ffmpeg libsm6 libxext6 libglib2.0-0 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="$CUDA_HOME/bin:$PATH"
ENV LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# 配置pip源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com && \
    pip install --upgrade pip setuptools wheel

RUN pip uninstall -y torch torchvision torchaudio
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://mirrors.nju.edu.cn/pytorch/whl/cu111

# 安装基础依赖
RUN pip uninstall -y numpy scikit-image pandas matplotlib shapely 2>/dev/null || true && \
    pip install --no-cache-dir \
    numpy==1.22.4 scikit-image==0.19.3 pandas==1.4.4 matplotlib==3.6 \
    shapely==1.8.5.post1 scikit-learn pyquaternion cachetools descartes future tensorboard IPython

# 安装open3d
RUN pip install --no-cache-dir open3d==0.19.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 复制项目依赖
COPY requirements.txt ./



# 安装项目依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torchmetrics==0.6.2

# 检查requirements.txt中是否有torch相关依赖
# RUN echo "检查requirements.txt是否包含torch..." && \
#    grep -i torch requirements.txt || echo "✓ 未检测到torch相关依赖"

RUN pip install openmim

ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA="1"
# 再安装其他库

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 9.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html && \
    pip install mmdet==2.14.0   && \
    pip install mmsegmentation==0.14.1 
   # sed -i '413s/raise RuntimeError/# raise RuntimeError/' /opt/venv/lib/python3.8/site-packages/torch/utils/cpp_extension.py && \
   # FORCE_CUDA=1 TORCH_CUDA_VERSION=11.1 pip install --no-cache-dir mmdet3d==0.17.1

RUN rm -rf /project/mmdetection3d && \
    git clone --depth 1 -b v0.17.1 https://github.com/open-mmlab/mmdetection3d.git /project/mmdetection3d && \
    cd /project/mmdetection3d && \
    CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 TORCH_CUDA_VERSION=11.1 python setup.py develop

RUN pip install torchattacks && \
    pip install --no-cache-dir mmengine==0.10.4 
# 最终验证
RUN python -c "import torch; import mmcv; import mmdet; import mmdet3d; print('=== 最终验证 ==='); print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}'); print(f'mmcv: {mmcv.__version__}'); print(f'mmdet: {mmdet.__version__}'); print(f'mmdet3d: {mmdet3d.__version__}')"

# 复制项目文件
COPY . ./

RUN ln -sf /usr/bin/python3 /usr/bin/python
    
CMD ["python", "main.py"]