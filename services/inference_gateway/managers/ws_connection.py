from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from websockets import connect, ClientConnection
from websockets.exceptions import InvalidHandshake, InvalidStatus, ConnectionClosed
import asyncio
from pydantic import BaseModel
import json
from uuid import UUID
from ..logger import logger
from ..data_models.chat_models import ChatResponse
from ..data_models.inference_models import LLMResponse


class ClientPingManager:
    def __init__(self, time_ping_s: int, pong_timeout_s: int):
        self.time_ping_s = time_ping_s
        self.pong_timeout_s = pong_timeout_s
        self.last_pong = {}
        self.ping_tasks = {}
    
    def get_frozen_client(self) -> list | None:
        if len(self.ping_tasks) > 0:
            wss = [ws for ws in self.ping_tasks if self.ping_tasks[ws].done()]
            for ws in wss:
                self.ping_tasks.pop(ws, None)
            return wss
        return None

    async def handle_pong(self, ws: WebSocket) -> None:
        self.last_pong[ws] = asyncio.get_running_loop().time()

    async def check_pong(self, ws: WebSocket):
        await asyncio.sleep(self.pong_timeout_s)

        if asyncio.get_running_loop().time() - self.last_pong.get(ws, 0) > self.pong_timeout_s:
            ws_info = f"{ws.client[0]}:{ws.client[1]}" if ws.client is not None else "unknown"
            logger.info(f"Клиент {ws_info} не отвечает")
            return False
        
        return True

    async def ping_client(self, ws: WebSocket) -> None:
        try:
            while True:
                await asyncio.sleep(self.time_ping_s)
                if ws.client_state != WebSocketState.CONNECTED:
                    break

                try:
                    await ws.send_json({"type": "ping"})
                    check_pong_task = asyncio.create_task(self.check_pong(ws))
                    ok = await check_pong_task
                    if not ok:
                        await self.remove_client(ws)
                        break

                except (WebSocketDisconnect, RuntimeError):
                    await self.remove_client(ws)
                    break

        except asyncio.CancelledError:
            pass
    
    def add_client(self, ws: WebSocket) -> None:
        task = asyncio.create_task(self.ping_client(ws))
        self.ping_tasks[ws] = task

    async def remove_client(self, ws: WebSocket) -> None:
        task = self.ping_tasks.get(ws)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class ClientConnectionManager:
    def __init__(self, client_ping_manager: ClientPingManager):
        self._active_connections = {}
        self._uuid_to_ws = {}
        self.disconnect_lock = asyncio.Lock()
        self.client_ping_manager = client_ping_manager
    
    @property
    def active_connections(self):
        return self._active_connections

    def pop_uuid_to_ws(self, uuid: UUID):
        return self._uuid_to_ws.pop(uuid, None)

    def get_conn_info(self, ws: WebSocket) -> str:
        if ws.client is None:
            return "unknown"
        host, port = ws.client
        return f"{host}:{port}"
    
    async def add_uuid(self, ws: WebSocket, uuid: UUID):
        if ws in self._active_connections:
            self._active_connections[ws].append(uuid)
            self._uuid_to_ws[uuid] = ws
        else:
            raise WebSocketDisconnect
    
    async def connect(self, ws: WebSocket) -> None:
        try:
            await ws.accept()
            self._active_connections[ws] = []
            self.client_ping_manager.add_client(ws)
            logger.info(f"Установлено WebSocket-соединение с клиентом {self.get_conn_info(ws)}")

        except RuntimeError as e:
            logger.error(f"Произошла ошибка при попытке принять WebSocket-соединение: {e}")
            raise RuntimeError
        
        except WebSocketDisconnect as e:
            logger.error("Клиент отсоединился во время принятия WebSocket-соединения")
            raise WebSocketDisconnect

    async def disconnect(self, ws: WebSocket, 
                         code: int = 1000) -> None:
        async with self.disconnect_lock:
            if ws in self._active_connections:
                try:
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.close(code)
                    
                except RuntimeError as e:
                    logger.error(f"Произошла ошибка при попытке разрыва соединения с клиентом {self.get_conn_info(ws)}: {e}")
                
                except WebSocketDisconnect as e:
                    logger.error(f"Произошла ошибка при попытке разрыва соединения с клиентом {self.get_conn_info(ws)}: Клиент уже отключен")

                finally:  
                    self._active_connections.pop(ws, None)
                    for uuid in list(self._uuid_to_ws):
                        if self._uuid_to_ws[uuid] == ws:
                            self.pop_uuid_to_ws(uuid)
                    logger.info(f"Разорвано WebSocket-соединение с клиентом {self.get_conn_info(ws)}: {code}")


class ClientManager:
    def __init__(self, time_ping_s: int, pong_timeout_s: int):
        self.client_ping_manager = ClientPingManager(time_ping_s=time_ping_s, 
                                                    pong_timeout_s=pong_timeout_s)
        self.client_connection_manager = ClientConnectionManager(self.client_ping_manager)
        self.pong_timeout_s = pong_timeout_s
        self.task_check_frozen_clients = asyncio.create_task(self.check_frozen_clients())

    async def stop(self) -> None:
        await asyncio.gather(*[self.client_connection_manager.disconnect(ws) 
                               for ws in self.client_connection_manager.active_connections
                               ])
        try:
            self.task_check_frozen_clients.cancel()

        except asyncio.CancelledError:
            pass
    
    async def check_frozen_clients(self):
        try:
            while True:
                await asyncio.sleep(self.pong_timeout_s + 5)
                wss = self.client_ping_manager.get_frozen_client()
                if wss is not None:
                    await asyncio.gather(*[self.client_connection_manager.disconnect(ws, 1001) for ws in wss])

        except asyncio.CancelledError as e:
            pass
    
    async def connect(self, ws: WebSocket):
        await self.client_connection_manager.connect(ws)

    async def disconnect(self, ws: WebSocket):
        await self.client_connection_manager.disconnect(ws)

    async def add_uuid(self, ws: WebSocket, uuid: UUID):
        await self.client_connection_manager.add_uuid(ws, uuid)

    async def handle_pong(self, ws: WebSocket):
        await self.client_ping_manager.handle_pong(ws)
    
    async def send_response(self, response: ChatResponse) -> None:
        ws = self.client_connection_manager.pop_uuid_to_ws(response.uuid)

        if ws is None:
            return

        try:
            await ws.send_json(response.model_dump(mode="json"))

        except WebSocketDisconnect as e:
            logger.error(f"Не удалось отправить запрос по WebSocket-соединению клиенту {self.client_connection_manager.get_conn_info(ws)}. " 
                         f"Причина: Клиент {self.client_connection_manager.get_conn_info(ws)} отсоединился")
            await self.client_connection_manager.disconnect(ws)
        
        except RuntimeError as e:
            logger.error(f"Произошла ошибка во время отправки ответа клиенту {self.client_connection_manager.get_conn_info(ws)}: {e}")


class ServiceConnectionManager:
    def __init__(self, max_delay: int = 60):
        self.active_connections = {}
        self.max_delay = max_delay
    
    def get_conn_info(self, endpoint: str) -> tuple[ClientConnection, str, int]:
        if endpoint not in self.active_connections:
            raise ValueError(f"Неизвестное WebSocket-соединение с {endpoint}")
        
        ws, host, port = self.active_connections[endpoint]
        return ws, host, port
    
    async def stop(self) -> None:
        await asyncio.gather(*[self.disconnect(endpoint) for endpoint in list(self.active_connections)])

    async def connect(self, endpoint: str, port: int, host: str) -> ClientConnection:
        delay = 1
        while delay < self.max_delay:
            try:
                ws = await connect(f"ws://{host}:{port}/{endpoint}", ping_interval=None, ping_timeout=None)
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
                delay = min(60, delay * 2)
                logger.warning(f"Не удалось установить WebSocket-соединение с {endpoint}: Переподключение через {delay} с")
                await asyncio.sleep(delay)
                continue

        raise ConnectionClosed

    async def disconnect(self, endpoint: str, code: int = 1000) -> None:
        try:
            ws, host, port = self.get_conn_info(endpoint)
            await ws.close(code=code)

            self.active_connections.pop(endpoint, None)
            logger.info(f"WebSocket-соединение с ws://{host}:{port}/{endpoint} разорвано: {code}")

        except ValueError as e:
            pass

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
                    try:
                        ws = await self.connect(endpoint, port, host)
                    except ConnectionClosed as e:
                        raise e
                    
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
                    try:
                        ws = await self.connect(endpoint, port, host)
                    except ConnectionClosed as e:
                        raise e
                    
                    logger.info(f"WebSocket-cоединение с инференс-сервисом ws://{host}:{port}/{endpoint} восстановлено")
                    continue
        except ValueError as e:
            logger.error(f"Попытка получить запрос по неизвестному WebSocket-соединению с {endpoint}")
            raise e

        return responses