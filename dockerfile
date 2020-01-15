ARG UBUNTU_VERSION=16.04
ARG CUDA=10.0
FROM nvidia/cuda:${CUDA}-base-ubuntu${UBUNTU_VERSION} as base
ARG CUDA=10.0
ARG CUDADASH=10-0
ARG CUDNN=7.4.1.5-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDADASH} \
        cuda-cublas-${CUDADASH} \
        cuda-cufft-${CUDADASH} \
        cuda-curand-${CUDADASH} \
        cuda-cusolver-${CUDADASH} \
        cuda-cusparse-${CUDADASH} \
        curl \
        libcudnn7=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        wget
RUN apt-get update && \
        apt-get install nvinfer-runtime-trt-repo-ubuntu1604-5.0.2-ga-cuda${CUDA} \
        && apt-get update \
        && apt-get install -y --no-install-recommends libnvinfer5=5.0.2-1+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ARG PYTHON=python3.6
ENV LANG C.UTF-8
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y ${PYTHON}
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN ${PYTHON} get-pip.py
RUN ln -sf /usr/bin/${PYTHON} /usr/local/bin/python3
RUN ln -sf /usr/local/bin/pip /usr/local/bin/pip3
RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python
ARG TF_PACKAGE=tensorflow-gpu==1.14.0
RUN pip3 install --upgrade ${TF_PACKAGE}
RUN apt-get update -qq -y \
  && apt-get install -y libsm6 libxrender1 libxext-dev python3-tk ffmpeg git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y python3.6-dev
COPY requirements.txt /opt/
RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install -r /opt/requirements.txt && rm /opt/requirements.txt
#patch for tensorflow:latest-gpu-py3 image
RUN cd /usr/local/cuda/lib64 \
  && mv stubs/libcuda.so ./ \
  && ln -s libcuda.so libcuda.so.1 \
  && ldconfig
RUN mkdir /app
WORKDIR /app
COPY . /app
EXPOSE 6006
ENTRYPOINT ["tail", "-f", "/dev/null"]
