FROM python:3.11-slim

WORKDIR /app

COPY . .
RUN mkdir /model && \
    mv /app/cached_model/* /model/ && \
    rm -rf /app/cached_model && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python3", "src/inference.py"]