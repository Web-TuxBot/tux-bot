from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect
import asyncio
import logging
import json
from datetime import datetime
from .data_models import ChatRequest, ChatResponse, LLMRequest, LLMResponse
from uuid import UUID


class ConnectionManager:
    def __init__(self):
        self.active_connections: set[WebSocket] = set()
        self.uuid_to_ws: dict[str, WebSocket] = {}
        logger.debug("Инициализирован менеджер соединений")
    
    def ws_info(self, ws: WebSocket) -> str:
        host, port = ws.client
        return f"{host}:{port}"

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active_connections.add(ws)
        logger.info(f"Установлено WebSocket-соединение с клиентом {self.ws_info(ws)}")

    def add_uuid(self, ws: WebSocket, uuid: UUID) -> None:
        if ws in self.active_connections:
            self.uuid_to_ws[uuid] = ws
        else:
            logger.error(f"""Не удалось добавить запрос с чата (uuid: {uuid}). 
                         Причина: Несуществующее WebSocket-соединение с клиентом {self.ws_info(ws)}.
                         Доступные WebSocket-соединения с клиентами: {'\n'.join(self.ws_info(ws) for ws in self.active_connections)}""", exc_info=True)
            raise ValueError(f"Несуществующее WebSocket-соединение с клиентом {self.ws_info(ws)}")

    async def disconnect(self, ws: WebSocket, code: int = 1000) -> None:
        if ws in self.active_connections:
            self.active_connections.discard(ws)
            if ws.client_state.name == "CONNECTED":
                await ws.close(code)
            uuids_to_remove = [uuid for uuid, sock in self.uuid_to_ws.items() if sock == ws]
            for uuid in uuids_to_remove:
                    del self.uuid_to_ws[uuid]
            logger.info(f"Разорвано WebSocket-соединение с клиентом {self.ws_info(ws)} (код: {code})")
        else:
            logger.error(f"""Попытка разрыва несуществующего WebSocket-соединения с клиентом {self.ws_info(ws)}.
                         Доступные WebSocket-соединения с клиентами: {'\n'.join(id(ws) for ws in self.active_connections)}""", exc_info=True)
    
    async def send_response(self, response: ChatResponse) -> None:
        try:
            logger.debug(f"Попытка отправить ответ, ключи: {self.uuid_to_ws.keys()}")
            ws = self.uuid_to_ws[response.uuid]
            await asyncio.wait_for(ws.send_json(response.model_dump(mode="json")), timeout=30)

        except asyncio.TimeoutError:
            logger.error(f"""Не удалось отправить запрос по WebSocket-соединению клиенту {self.ws_info(ws)}. 
                        Причина: Превышено время ожидания (Timeout)""", exc_info=True)
            await self.disconnect(ws, 1006)
            
        except WebSocketDisconnect:
            logger.error(f"""Не удалось отправить запрос по WebSocket-соединению клиенту {self.ws_info(ws)}. 
                        Причина: Клиент {self.ws_info(ws)} отсоединился""", exc_info=True)
            await self.disconnect(ws)
            

class InferenceManager:
    def __init__(self, model_name: str, max_buffer_len: int = 10, interval_time_ms: int = 100):
        self.inference_uri = f"inference/{model_name}/generate"
        self.max_buffer_len: int = max_buffer_len
        self.interval_time_s: float = interval_time_ms / 1000
        self.lock: asyncio.Lock = asyncio.Lock()
        self.buffer: dict[str, str] = {}
        self.batch_queue: asyncio.Queue = asyncio.Queue()
        self.pending: dict[str, asyncio.Future] = {}
        logger.debug(f"""Инициализирован менеджер инференса.
                        Параметры:
                        Inference URI: {self.inference_uri}
                        Длина буфера: {self.max_buffer_len}
                        Время сбора батча: {self.interval_time_s} с""")

    async def add_request(self, uuid: str, message: str) -> asyncio.Future:
        async with self.lock:
            self.pending[uuid] = asyncio.get_event_loop().create_future()
            self.buffer[uuid] = message

            if len(self.buffer) == self.max_buffer_len:
                self.batch_queue.put_nowait(dict(self.buffer))
                self.buffer.clear()
        
        return self.pending[uuid]
    
    async def connect_inference(self):
        self.ws_inference = await connect(f"ws://localhost:8002/{self.inference_uri}", ping_interval=None, ping_timeout=None)
    
    async def inference(self, batch: dict) -> tuple[dict[str, list[str] | str], str]:
        requests, uuids = [], []

        for uuid, request in batch.items():
            requests.append(request)
            uuids.append(uuid)
        requests = LLMRequest(requests=requests)

        await self.ws_inference.send(requests.model_dump_json())
        responses = json.loads(await self.ws_inference.recv())
        responses = LLMResponse(**responses)
        created_at = responses.created_at
        
        responses_dict = {}
        for response, uuid in zip(responses.responses, uuids):
            responses_dict[uuid] = response 

        return responses_dict, created_at
    
    async def inference_worker(self) -> None:
        while True:
            await asyncio.sleep(self.interval_time_s)
            async with self.lock:
                batch = []
                if not self.batch_queue.empty():
                    batch = await self.batch_queue.get()
                elif self.buffer:
                    batch = self.buffer.copy()
                    self.buffer.clear()
                else:
                    continue
            
            responses, created_at = await self.inference(batch)

            for uuid, response in responses.items():
                future = self.pending.pop(uuid, None)
                if future and not future.done():
                    future.set_result((response, created_at))


async def lifespan(app: FastAPI):
    asyncio.create_task(app.state.inference_manager.inference_worker())
    await app.state.inference_manager.connect_inference()
    yield


def init_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    log_date = datetime.now().strftime("%Y-%m-%d")
    file_handler = logging.FileHandler(f"services/logs/inference_gateway_{log_date}.log")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
    

def create_app():
    global logger
    logger = init_logger()
    app = FastAPI(lifespan=lifespan)
    app.state.inference_manager = InferenceManager("qwen2-5_instruct")
    app.state.conn_manager = ConnectionManager()

    @app.websocket("/inference/batching")
    async def websocket_endpoint(ws: WebSocket):
        await app.state.conn_manager.connect(ws)
        host, port = ws.client

        try:
            while True:
                try:
                    data = await asyncio.wait_for(ws.receive_json(), timeout=30)
                except asyncio.TimeoutError:
                    logger.error(f"Соединение с клиентом {host}:{port} потеряно")
                    await app.state.conn_manager.disconnect(ws, code=1000)
                    break

                req = ChatRequest(**data)

                try:
                    app.state.conn_manager.add_uuid(ws, req.uuid)
                except ValueError:
                    await app.state.conn_manager.disconnect(ws)
                    break

                fut = await app.state.inference_manager.add_request(req.uuid, req.message)

                async def handle_request(uuid: str, fut: asyncio.Future) -> None:
                    response, created_at = await fut
                    response = ChatResponse(uuid=uuid, response=response)
                    await app.state.conn_manager.send_response(response)

                asyncio.create_task(handle_request(req.uuid, fut))

        except WebSocketDisconnect:
            await app.state.conn_manager.disconnect(ws, code=1000)
    
    return app
