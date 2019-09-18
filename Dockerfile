# Taken from Satish's repo
## commands to build and run this docker file
# docker build -t deep-conv-rf:latest - < Dockerfile
# docker run -it --rm -v <local_host_dir_path>:/root/workspace/ --name deep-conv-rf-env deep-conv-rf:latest

FROM ubuntu:18.04

# set maintainer
LABEL maintainer="bvarjav1@jhu.edu"

# System Setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
	&& apt-get install -y vim \
        cmake \
        cpio \
        gfortran \
        libpng-dev \
        freetype* \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        software-properties-common\
        git \
        man \
        wget \
		openssl \
		libssl-dev \
		libcurl4-openssl-dev \
		git \
		gpg \
		libomp-dev \
		libeigen3-dev \
		gsl-bin \
		libgslcblas0 \
		libgsl23 \
		libgsl-dev \
		libxml2-dev

# update
RUN apt-get update && apt-get -y upgrade

RUN apt-get install -y python3-venv python3-pip python3-dev build-essential cmake
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip3 install pandas

# Install RerF dependencies
RUN apt-get install -y build-essential cmake python3-dev libomp-dev vim

# make a directory for mounting local files into docker
RUN mkdir /root/workspace/
# change working directory to install RerF
RUN mkdir /root/code/
WORKDIR /root/code/

# clone the RerF code into the container
RUN git clone https://github.com/neurodata/RerF.git .

# go to Python subdir (install python bindings)
WORKDIR /root/code/Python

# install python requirements
RUN pip3 install -r requirements.txt
RUN pip3 install matplotlib seaborn pandas jupyter pycodestyle torch torchvision pytest scikit-learn scipy

# clean old installs
RUN python3 setup.py clean --all

# install RerF
RUN pip3 install -e .

# add RerF to PYTHONPATH for dev purposes
RUN echo "export PYTHONPATH='${PYTHONPATH}:/root/code'" >> ~/.bashrc

# clean dir and test if RerF is correctly installed
RUN py3clean .

# set working dir to workspace (you can mount any local host dir into this path)
WORKDIR /root/workspace/

# launch terminal
CMD ["/bin/bash"]
