from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from omegaconf import OmegaConf
from .inference_gateway.data_models.inference_models import LLMResponse, LLMRequest
from pathlib import Path
from datetime import datetime
import asyncio
import logging
import os


class LLMModel:
    def __init__(self, model_name: str):
        self.cls = self._get_model_class(model_name)
        self.cfg = self._get_model_config(model_name)
        self.model = self.cls(self.cfg)
        self.model.load_model()

    def _get_model_class(self, model_name: str):
        if model_name == "qwen2-5_instruct":
            from models.qwen_modeling import Qwen2_5Instruct
            return Qwen2_5Instruct
        else:
            raise ValueError(f"Модель {model_name} не поддерживается")
    
    def _get_model_config(self, model_name: str):
        cfg_name = f"{model_name}_config.yaml"
        cfg_path = Path(__file__).parent.parent / f"model_configs/{cfg_name}"
        cfg = OmegaConf.load(cfg_path)
        return cfg
    
    def get_response(self, batch: list[str]):
        requests = self.model.generate_response(batch)
        return requests


async def lifespan(app: FastAPI):
    #app.state.model = LLMModel(model_name=app.state.model_name)
    yield


def init_logger(model_name: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    log_date = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(f"services/logs/inference_{model_name}_{log_date}.log")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def create_app():
    app = FastAPI(lifespan=lifespan)
    app.state.model_name = os.environ.get("MODEL_NAME")
    logger = init_logger(app.state.model_name)

    @app.websocket(f"/inference/generate")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        host, port = ws.client
        responses = []
        try:
            while True:
                try:
                    data = await asyncio.wait_for(ws.receive_json(), timeout=300)
                except asyncio.TimeoutError:
                    logger.critical(f"Соединение с клиентом {host}:{port} потеряно при попытке принять запрос")
                    await ws.close()
                    break
                batch = LLMRequest(**data)
                # ЭТО ЭХО ЗАГЛУШКА!!!
                for i in range(len(batch.requests)):
                    responses.append(batch.requests[i])
                ########################################
                #responses = app.state.model.get_response(batch.requests)
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

    


    