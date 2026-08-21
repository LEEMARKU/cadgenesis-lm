# CADGenesis-LM training image (Pillar 11 deployment)
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml setup.py requirements.txt ./
COPY src ./src
COPY sdk ./sdk

RUN pip install --no-cache-dir -e ".[serve]"

ENV PYTHONPATH=/app/src:/app/sdk
ENV CADGENESIS_OUTPUT_DIR=/outputs

RUN mkdir -p /outputs && \
    useradd --create-home --uid 1000 cadgenesis && \
    chown -R cadgenesis:cadgenesis /outputs /app

USER cadgenesis

VOLUME ["/outputs"]

CMD ["python", "-m", "cadgenesis.cli", "train", "--config", "configs/train.yaml"]
