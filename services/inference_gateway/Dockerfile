FROM python:3.13-slim
LABEL title="Inference Gateway"
LABEL maintainer="artempolyakov159@gmail.com"
LABEL description="Inference Gateway for backend and ineference communication"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && mkdir inference_gateway
COPY . /app/inference_gateway

ENV INFERENCE_SERVICES='[{"model_name": "qwen2-5_instruct", "host": "0.0.0.0", "port": 8002, "max_buffer_len": 50, "interval_time_ms": 100}]'
ENV CLIENT_TIME_PING_S=30
ENV CLIENT_TIME_PONG_S=25

ENTRYPOINT ["uvicorn", "--factory", "inference_gateway.app:create_app"]
CMD ["--port", "8000", "--log-level", "info", "--host", "0.0.0.0", "--workers", "1"]


