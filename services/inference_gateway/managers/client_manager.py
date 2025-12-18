from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import asyncio
from uuid import UUID
from ..logger import logger
from ..data_models.chat_models import ChatResponse


class ClientPingManager:
    def __init__(self, ping_interval: int, ping_timeout: int):
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.ping_tasks = {}
        self.pong_events = {}
        self._frozen_clients = []
    
    async def stop(self):
        for task in self.ping_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @property
    def frozen_clients(self):
        if len(self._frozen_clients) > 0:
            return self._frozen_clients
        else:
            return None
        
    def delete_frozen_client(self, ws: WebSocket):
        if ws in self._frozen_clients:
            ws_idx = self._frozen_clients.index(ws)
            self._frozen_clients.pop(ws_idx)

    async def handle_pong(self, ws: WebSocket) -> None:
        if ws in self.pong_events:
            self.pong_events[ws].set()
        else:
            raise WebSocketDisconnect

    async def ping_client(self, ws: WebSocket) -> None:
        try:
            while True:
                await asyncio.sleep(self.ping_interval)
                if ws.client_state != WebSocketState.CONNECTED:
                    self.remove_client(ws)
                    break

                try:
                    await ws.send_json({"type": "ping"})
                    await asyncio.wait_for(self.pong_events[ws].wait(), timeout=self.ping_timeout)

                except (WebSocketDisconnect, RuntimeError) as e:
                    await self.remove_client(ws)
                    break

                except asyncio.TimeoutError as e:
                    ws_info = f"{ws.client[0]}:{ws.client[1]}" if ws.client is not None else "unknown"
                    logger.info(f"Клиент {ws_info} не отвечает")
                    self._frozen_clients.append(ws)
                    await self.remove_client(ws)
                    break

        except asyncio.CancelledError:
            pass
    
    def add_client(self, ws: WebSocket) -> None:
        self.pong_events[ws] = asyncio.Event()
        task = asyncio.create_task(self.ping_client(ws))
        self.ping_tasks[ws] = task

    async def remove_client(self, ws: WebSocket) -> None:
        self.pong_events.pop(ws, None)
        task = self.ping_tasks.pop(ws, None)
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
                    await self.client_ping_manager.remove_client(ws)
                    logger.info(f"Разорвано WebSocket-соединение с клиентом {self.get_conn_info(ws)}: {code}")


class ClientManager:
    def __init__(self, ping_interval: int, ping_timeout: int):
        self.client_ping_manager = ClientPingManager(ping_interval=ping_interval, 
                                                    ping_timeout=ping_timeout)
        self.client_connection_manager = ClientConnectionManager(self.client_ping_manager)
        self.ping_timeout = ping_timeout
        self.task_check_frozen_clients = asyncio.create_task(self.check_frozen_clients())

    async def stop(self) -> None:
        await asyncio.gather(*[self.client_connection_manager.disconnect(ws) 
                               for ws in list(self.client_connection_manager.active_connections.keys())
                               ])
        try:
            self.task_check_frozen_clients.cancel()
            await self.client_ping_manager.stop()

        except asyncio.CancelledError:
            pass
    
    async def check_frozen_clients(self):
        try:
            while True:
                await asyncio.sleep(self.ping_timeout)
                wss = self.client_ping_manager.frozen_clients
                if wss is not None:
                    await asyncio.gather(*[self.client_connection_manager.disconnect(ws, 1001) for ws in wss])
                    for ws in wss:
                        self.client_ping_manager.delete_frozen_client(ws)

        except asyncio.CancelledError as e:
            pass

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
    
    async def connect(self, ws: WebSocket):
        try:
            await self.client_connection_manager.connect(ws)
            
        except RuntimeError as e:
            raise e
        
        except WebSocketDisconnect as e:
            raise e

    async def disconnect(self, ws: WebSocket):
        try:
            await self.client_connection_manager.disconnect(ws)

        except RuntimeError as e:
            raise e
        
        except WebSocketDisconnect as e:
            raise e

    async def add_uuid(self, ws: WebSocket, uuid: UUID):
        try:
            await self.client_connection_manager.add_uuid(ws, uuid)
            
        except WebSocketDisconnect as e:
            raise e

    async def handle_pong(self, ws: WebSocket):
        try:
            await self.client_ping_manager.handle_pong(ws)

        except WebSocketDisconnect as e:
            await self.client_ping_manager.remove_client(ws)
            raise e
