FROM nvidia/cuda:12.1.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=nointeractive

RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0 \
        build-essential \
        libssl-dev

RUN python3 -m pip install --upgrade pip

RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

WORKDIR /app

# Set the entrypoint
ENTRYPOINT [ "python3" ]