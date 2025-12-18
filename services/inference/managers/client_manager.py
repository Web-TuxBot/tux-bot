from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from starlette.websockets import WebSocketState
import asyncio
from ..logger import logger


class ClientConnectionManager:
    def __init__(self, max_connection: int):
        self._active_connections = []
        self.max_connection = max_connection
        self.disconnect_lock = asyncio.Lock()
    
    @property
    def active_connections(self):
        return self._active_connections
    
    def get_conn_info(self, ws: WebSocket) -> str:
        if ws.client is None:
            return "unknown"
        host, port = ws.client
        return f"{host}:{port}"
    
    async def connect(self, ws: WebSocket) -> None:
        try:
            if len(self._active_connections) == self.max_connection:
                await ws.close(1008)
                logger.error(f"Отказано в доступе клиенту {self.get_conn_info(ws)}: Превышено число клиентов")
                raise WebSocketDisconnect
            
            await ws.accept()
            self._active_connections.append(ws)
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
                        self._active_connections.remove(ws)
                        logger.info(f"Разорвано WebSocket-соединение с клиентом {self.get_conn_info(ws)}: {code}")
                    
                except RuntimeError as e:
                    logger.error(f"Произошла ошибка при попытке разрыва соединения с клиентом {self.get_conn_info(ws)}: {e}")
                
                except WebSocketDisconnect as e:
                    logger.error(f"Произошла ошибка при попытке разрыва соединения с клиентом {self.get_conn_info(ws)}: Клиент уже отключен")


class ClientManager:
    def __init__(self, max_connection: int):
        self.client_connection_manager = ClientConnectionManager(max_connection=max_connection)

    async def stop(self) -> None:
        await asyncio.gather(*[self.client_connection_manager.disconnect(ws) 
                               for ws in self.client_connection_manager.active_connections.copy()
                               ])

    async def send_response(self, response: BaseModel, ws: WebSocket) -> None:
        if ws is None:
            return

        try:
            await ws.send_json(response.model_dump(mode="json"))

        except WebSocketDisconnect as e:
            logger.error(f"Не удалось отправить ответы по WebSocket-соединению клиенту {self.client_connection_manager.get_conn_info(ws)}. " 
                         f"Причина: Клиент {self.client_connection_manager.get_conn_info(ws)} отсоединился")
            raise e
        
        except RuntimeError as e:
            logger.error(f"Произошла ошибка во время отправки ответа клиенту {self.client_connection_manager.get_conn_info(ws)}: {e}")
            raise WebSocketDisconnect
    
    async def connect(self, ws: WebSocket):
        try:
            await self.client_connection_manager.connect(ws)
        except WebSocketDisconnect as e:
            raise e
        except RuntimeError as e:
            raise e

    async def disconnect(self, ws: WebSocket):
        try:
            await self.client_connection_manager.disconnect(ws)
        except WebSocketDisconnect as e:
            raise e
        except RuntimeError as e:
            raise e