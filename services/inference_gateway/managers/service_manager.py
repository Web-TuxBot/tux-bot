from websockets import connect, ClientConnection
from websockets.exceptions import InvalidHandshake, InvalidStatus, ConnectionClosed
import asyncio
import json
from ..logger import logger


class ServiceConnectionManager:
    def __init__(self, max_delay: int, ping_interval: int, ping_timeout: int):
        self.active_connections = {}
        self.max_delay = max_delay
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
    
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
                ws = await connect(f"ws://{host}:{port}/{endpoint}", 
                                   ping_interval=self.ping_interval, ping_timeout=self.ping_timeout)
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

    async def safe_send(self, endpoint: str, requests: dict) -> None:
        try:
            ws, host, port = self.get_conn_info(endpoint)
            while True:
                try:
                    await ws.send(json.dumps(requests))
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

    async def safe_recv(self, endpoint: str, requests: dict) -> dict:
        try:
            ws, host, port = self.get_conn_info(endpoint)
            while True:
                try:
                    responses = json.loads(await ws.recv())
                    break
                except ConnectionClosed as e:
                    logger.warning(f"Потеряно WebSocket-соединение с инференс-сервисом ws://{host}:{port}/{endpoint} при приеме ответов: Попытка переподключения")
                    try:
                        ws = await self.connect(endpoint, port, host)
                    except ConnectionClosed as e:
                        raise e
                    
                    logger.info(f"WebSocket-cоединение с инференс-сервисом ws://{host}:{port}/{endpoint} восстановлено")
                    try:
                        await self.safe_send(endpoint, requests)
                    except ConnectionClosed as e:
                        raise e
                    continue
                
        except ValueError as e:
            logger.error(f"Попытка получить запрос по неизвестному WebSocket-соединению с {endpoint}")
            raise e

        return responses