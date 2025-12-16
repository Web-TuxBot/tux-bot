FROM python:3.13-slim
LABEL title="Test Inference Echo"
LABEL maintainer="artempolyakov159@gmail.com"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && mkdir inference
COPY . /app/inference

ENV MODEL_NAME="qwen2-5_instruct"

ENTRYPOINT ["uvicorn", "--factory", "inference.inference:create_app"]
CMD ["--port", "8002", "--log-level", "info", "--host", "0.0.0.0", "--workers", "1"]


