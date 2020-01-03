
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

LABEL maintainer Ilija Vukotic <ivukotic@cern.ch>

###################
#### CUDA stuff
###################
RUN echo "/usr/local/cuda-10.2/lib64/" >/etc/ld.so.conf.d/cuda.conf

# For CUDA profiling, TensorFlow requires CUPTI.
RUN echo "/usr/local/cuda/extras/CUPTI/lib64/" >>/etc/ld.so.conf.d/cuda.conf

# make sure we have a way to bind host provided libraries
# see https://github.com/singularityware/singularity/issues/611
RUN mkdir -p /host-libs && \
    echo "/host-libs/" >/etc/ld.so.conf.d/000-host-libs.conf


#################
#### curl/wget
#################
RUN apt-get update && apt-get install curl wget -y

####################
#### Ubuntu packages
####################
# bazel is required for some TensorFlow projects
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" >/etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add -

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get install -y --allow-unauthenticated \
    build-essential \
    git \
    module-init-tools \
    pkg-config \
    python3 \
    python3-pip \
    rsync \
    software-properties-common \
    unzip \
    zip \
    vim \
    libopenmpi-dev \
    bazel \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN pip3 install --upgrade pip

RUN pip3 --no-cache-dir install \
    requests \
    elasticsearch \
    h5py \
    matplotlib \
    numpy \
    pandas 

RUN pip3 --no-cache-dir install \
    tables \
    scipy \
    sklearn \
    keras

RUN pip3 --no-cache-dir install \
    tqdm \
    tensorflow-gpu \
    gym \
    baselines 

COPY environment.sh /.environment.sh
COPY *.py /

# build info
RUN echo "Timestamp:" `date --utc` | tee /image-build-info.txt


CMD ["/.environment.sh"]