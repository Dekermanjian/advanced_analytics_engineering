ARG BASE_IMAGE=python:3.12.7-slim
FROM $BASE_IMAGE AS runtime-environment

# install base utils
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# put conda in path
ENV PATH=$CONDA_DIR/bin:$PATH

# install project requirements
COPY environment.yml /tmp/environment.yml
RUN conda env update --name base --file /tmp/environment.yml --prune