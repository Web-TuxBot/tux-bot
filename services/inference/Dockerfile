FROM python:3.13-slim
LABEL title="Inference"
LABEL maintainer="artempolyakov159@gmail.com"
LABEL description="Inference for generating responses"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && mkdir inference
COPY . /app/inference

ENV MODEL_NAME="qwen2-5_instruct"
ENV MAX_LEN_QUEUE=4
ENV MAX_CONNECTION=4

ENTRYPOINT ["uvicorn", "--factory", "inference.app:create_app"]
CMD ["--port", "8002", "--log-level", "info", "--host", "0.0.0.0"]


