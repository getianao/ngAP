FROM nvidia/cuda:12.0.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update 
RUN apt-get install -y libtbb-dev=2020.1-2 wget git
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /opt/cmake-3.24.1 \
      && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
      && rm /tmp/cmake-install.sh \
      && ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin
# hyperscan
RUN apt-get install -y ragel nasm libsqlite3-dev pkg-config
RUN apt-get install -y libboost-all-dev
RUN echo "deb http://dk.archive.ubuntu.com/ubuntu/ xenial main" |  tee -a /etc/apt/sources.list
RUN echo "deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe" |  tee -a /etc/apt/sources.list
RUN apt-get update && apt-get install -y g++-5 gcc-5 
RUN rm -rf /var/lib/apt/lists/*


# install python environment
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda init bash
RUN /bin/bash -c "source /root/.bashrc"
RUN conda install -y numpy scipy pandas seaborn -c conda-forge
RUN pip install https://github.com/getianao/figurePlotter/archive/refs/tags/v0.23.9.14.tar.gz

ENV NGAP_ROOT="/ngAP"
ENV PATH="/ngAP/code/build/bin:${PATH}"
ENV PATH="/ngAP/hscompile/build:${PATH}"

WORKDIR /ngAP