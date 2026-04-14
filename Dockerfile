ARG BASE_IMAGE=runpod/worker-v1-vllm:v2.11.3
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

COPY handler.py /workspace/handler.py

CMD ["python", "-u", "/workspace/handler.py"]
