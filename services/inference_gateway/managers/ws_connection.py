from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from websockets import connect, ClientConnection
from websockets.exceptions import InvalidHandshake, InvalidStatus, ConnectionClosed
import asyncio
from asyncio import Event
from pydantic import BaseModel
import json
from uuid import UUID
from ..logger import logger
from ..data_models.chat_models import ChatResponse
from ..data_models.inference_models import LLMResponse


class ClientConnectionManager:
    def __init__(self):
        self.active_connections = {}
        self.disconnected_uuids = set()
        self.uuid_to_ws = {}
        self.reconnect_events = []
        self.lock = asyncio.Lock()
        logger.debug("Инициализирован ClientConnectionManager")
    
    def get_conn_info(self, ws: WebSocket) -> str:
        if ws.client is None:
            return "unknown"
        host, port = ws.client
        return f"{host}:{port}"

    async def stop(self) -> None:
        logger.info("ClientConnectionManager завершает свою работу...")
        for ws in list(self.active_connections):
            await self.disconnect(ws)
        logger.info("ClientConnectionManager завершил свою работу")
    
    async def add_uuid(self, ws: WebSocket, uuid: UUID):
        if ws in self.active_connections:
            async with self.lock:
                self.active_connections[ws].append(uuid)
                self.uuid_to_ws[uuid] = ws
        else:
            raise WebSocketDisconnect
    
    async def get_reconnect_status(self, event_idx: int):
        reconect_event: Event = self.reconnect_events[event_idx][0]
        await reconect_event.wait()
        return self.reconnect_events[event_idx]

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active_connections[ws] = []
        logger.info(f"Установлено WebSocket-соединение с клиентом {self.get_conn_info(ws)}")

    async def _reconnect_timer(self, uuids: tuple[UUID], event_idx: int) -> None:
        reconect_event = self.reconnect_events[event_idx][0]
        await asyncio.sleep(2)
        if uuids in self.disconnected_uuids:
            async with self.lock:
                self.disconnected_uuids.discard(uuids)
                for uuid in uuids:
                    del self.uuid_to_ws[uuid]
            logger.warning("Не удалось переподключиться к клиенту")
        else:
            self.reconnect_events[event_idx][1] = True
            logger.warning("Переподключение к клиенту прошло успешно")
        reconect_event.set()
        return
    
    async def reconnect(self, ws: WebSocket, uuids: tuple[UUID]) -> bool:
        if uuids in self.disconnected_uuids:
            async with self.lock:
                self.active_connections[ws] = list(uuids)
                self.disconnected_uuids.discard(uuids)
        else:
            logger.warning(f"Не удалось восстановить запросы клиента {self.get_conn_info(ws)} после переподключения")

    async def disconnect(self, ws: WebSocket, code: int = 1000, reconnect: bool = False) -> int | None:
        if ws in self.active_connections:
            uuids = tuple(self.active_connections[ws])
            
            try:
                await ws.close(code)
            except RuntimeError:
                pass

            del self.active_connections[ws]
            logger.info(f"Разорвано WebSocket-соединение с клиентом {self.get_conn_info(ws)}: {code}")

            if reconnect:
                logger.warning(f"Попытка переподключения к клиенту...")
                self.disconnected_uuids.add(uuids)
                reconnect_event = Event()
                self.reconnect_events.append((reconnect_event, False))
                asyncio.create_task(self._reconnect_timer(uuids, len(self.reconnect_events) - 1))
                return len(self.reconnect_events) - 1
        else:
            logger.error(f"Попытка разрыва несуществующего WebSocket-соединения с клиентом {self.get_conn_info(ws)}."
                         f"Доступные WebSocket-соединения с клиентами: {'\n'.join(self.get_conn_info(ws) for ws in self.active_connections)}")
    
    async def send_response(self, response: ChatResponse) -> None:
        ws = self.uuid_to_ws.get(response.uuid)
        if ws is None:
            logger.warning(f"UUID {response.uuid} не найден - ответ не будет отправлен")
            return

        try:
            await ws.send_json(response.model_dump(mode="json"))

        except WebSocketDisconnect as e:
            logger.error(f"""Не удалось отправить запрос по WebSocket-соединению клиенту {self.get_conn_info(ws)}. 
                             Причина: Клиент {self.get_conn_info(ws)} отсоединился""")
            await self.disconnect(ws)
        
        except RuntimeError as e:
            pass


class ServiceConnectionManager:
    def __init__(self, max_delay: int = 60):
        self.active_connections = {}
        self.max_delay = max_delay
        logger.debug("Инициализирован ServiceConnectionManager")
    
    def get_conn_info(self, endpoint: str) -> tuple[ClientConnection, str, int]:
        if endpoint not in self.active_connections:
            raise ValueError(f"Неизвестное WebSocket-соединение с {endpoint}")
        
        ws, host, port = self.active_connections[endpoint]
        return ws, host, port
    
    async def stop(self) -> None:
        logger.info("Завершение работы ServiceConnectionManager...")
        for endpoint in list(self.active_connections):
            await self.disconnect(endpoint)
        logger.info("ServiceConnectionManager завершил свою работу")

    async def connect(self, endpoint: str, port: int, host: str) -> ClientConnection:
        delay = 1
        while delay < self.max_delay:
            try:
                ws = await connect(f"ws://{host}:{port}/{endpoint}", ping_interval=None, ping_timeout=None)
                client = ws.remote_address
                logger.info(f"Успешное подключение к сервису ws://{host}:{port}/{endpoint}")
                self.active_connections[endpoint] = (ws, host, port)
                return ws

            except InvalidHandshake as e:
                logger.critical(f"{endpoint} не принимает WebSocket-соединение")
                raise e
            
            except InvalidStatus as e:
                logger.critical(f"Не удалось установить WebSocket-соединение с {endpoint}: HTTP {e.response.status_code}")
                raise e
            
            except OSError as e:
                logger.warning(f"Не удалось установить WebSocket-соединение с {endpoint}: Переподключение через {delay} с")
                delay = min(60, delay * 2)
                await asyncio.sleep(delay)
                continue
        raise

    async def disconnect(self, endpoint: str, code: int = 1000) -> None:
        try:
            ws, host, port = self.get_conn_info(endpoint)
            await ws.close(code=code)

            del self.active_connections[endpoint]
            logger.info(f"WebSocket-соединение с ws://{host}:{port}/{endpoint} разорвано: {code}")

        except ValueError as e:
            logger.error(f"Попытка отправить разорвать неизвестное WebSocket-соединение с {endpoint}")

    async def safe_send(self, endpoint: str, requests: BaseModel) -> None:
        try:
            ws, host, port = self.get_conn_info(endpoint)
            while True:
                try:
                    await ws.send(requests.model_dump_json())
                    break
                except ConnectionClosed as e:
                    logger.warning(f"Потеряно WebSocket-соединение с сервисом ws://{host}:{port}/{endpoint}" 
                                   f"при отправке запроса: HTTP {e.rcvd.code} Переподключение...")
                    ws = await self.connect(endpoint, port, host)
                    logger.info(f"WebSocket-соединение с сервисом ws://{host}:{port}/{endpoint} восстановлено")
                    continue

        except ValueError as e:
            logger.error(f"Попытка отправить запрос по неизвестному WebSocket-соединению с {endpoint}")
            raise e

    async def safe_recv(self, endpoint: str) -> BaseModel:
        try:
            ws, host, port = self.get_conn_info(endpoint)
            while True:
                try:
                    responses = json.loads(await ws.recv())
                    responses = LLMResponse(**responses)
                    break
                except ConnectionClosed as e:
                    logger.warning(f"Потеряно WebSocket-соединение с инференс-сервисом ws://{host}:{port}/{endpoint} при приеме ответов: Попытка переподключения")
                    ws = await self.connect(endpoint, port, host)
                    logger.info(f"WebSocket-cоединение с инференс-сервисом ws://{host}:{port}/{endpoint} восстановлено")
                    continue
        except ValueError as e:
            logger.error(f"Попытка получить запрос по неизвестному WebSocket-соединению с {endpoint}")
            raise e

        return responses