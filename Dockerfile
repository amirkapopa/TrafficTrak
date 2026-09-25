# Offline evaluation image for an NVIDIA T4-class GPU (CUDA 12 + cuDNN 9, Python 3.10).
#   docker build -t traffictrak .
#   docker run --gpus all --network none -v /path/to/test_videos:/data/test -v $PWD/out:/out traffictrak
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-pip curl ca-certificates libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
# Weights are fetched at build time so the container runs with --network none.
RUN bash weights/download.sh

# Default = the organisers' official command.
CMD ["python3", "run_submission.py", "--videos", "/data/test", "--out", "/out/predictions.json", "--team", "TrafficTrak"]
