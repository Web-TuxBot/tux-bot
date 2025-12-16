from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from .inference_models import LLMResponse, LLMRequest
from datetime import datetime
import asyncio
import logging
import colorlog
import os


def init_logger():
    logger = logging.getLogger("inference")
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s %(levelname)s%(reset)s - %(message)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red,bg_white'
        },
        datefmt='%Y-%m-%d %H:%M:%C',
        reset=True,
        style='%' 
        )

    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger

# Говнокод, есть более новая версия инференса, это чисто для эхо
def create_app():
    app = FastAPI()
    model_name = os.getenv("MODEL_NAME")
    logger = init_logger()

    @app.websocket(f"/inference/{model_name}/generate")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        host, port = ws.client
        responses = []
        try:
            while True:
                try:
                    data = await ws.receive_json()
                except asyncio.TimeoutError:
                    logger.critical(f"Соединение с клиентом {host}:{port} потеряно при попытке принять запрос")
                    await ws.close()
                    break
                batch = LLMRequest(**data)
                for i in range(len(batch.requests)):
                    responses.append(batch.requests[i])
                created_at = datetime.now().isoformat()
                try:
                    await asyncio.wait_for(ws.send_json(
                        LLMResponse(responses=responses, created_at=created_at).model_dump()), timeout=30)
                    responses.clear()
                except asyncio.TimeoutError:
                    logger.critical(f"Соединение с клиентом {host}:{port} потеряно при попытке отправить ответ")
                    await ws.close()
                    break

        except WebSocketDisconnect:
            pass

    return app

    


    
