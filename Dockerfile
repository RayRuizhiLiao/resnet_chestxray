FROM ubuntu:18.04

MAINTAINER Ray Liao <ruizhi@mit.edu>

RUN apt-get update && apt-get install -y --no-install-recommends \
        ants \
        ca-certificates \
        curl \
        g++ \
        inotify-tools \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /opt/mlmodel

RUN pip3 install --no-cache-dir Cython

COPY requirements.txt /opt/mlmodel
RUN pip3 install --no-cache-dir -r /opt/mlmodel/requirements.txt

COPY . /opt/mlmodel
WORKDIR /opt/mlmodel
ENV PYTHONPATH=/opt/mlmodel:$PYTHONPATH

RUN chmod +x /opt/mlmodel/test_inference.py

ENTRYPOINT ["python3", "/opt/mlmodel/test_inference.py"]
