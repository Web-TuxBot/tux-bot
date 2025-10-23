from fastapi import FastAPI, WebSocket
import asyncio
from data_models import ChatRequest, ChatResponse
#TODO (mortiferr): Добавить логи и обработку исключений везде, где требуется

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[WebSocket, set[str]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = set()

    def disconnect(self, websocket: WebSocket):
        del self.active_connections[websocket]
    
    def add_uuid(self, websocket: WebSocket, uuid: str):
        if websocket in self.active_connections:
            self.active_connections[websocket].add(uuid)
    
    def search_websocket_by_uuid(self, uuid: str) -> WebSocket:
        for websocket in self.active_connections.keys():
            if uuid in self.active_connections[websocket]:
                return websocket
        return None
    
    async def send_response(self, response: ChatResponse):
        websocket = self.search_websocket_by_uuid(response.uuid)
        try:
            await websocket.send_json(response.model_dump())
        except Exception as e:
            print(f"Error: {e}")
            self.disconnect(websocket)
            
conn_manager = ConnectionManager()

#TODO(mortiferrr): Убрать временную заглушку инференса и реализовать подключение к инференс-сервису
class InferenceManager:
    def __init__(self, max_buffer_len: int = 10, interval_time_ms: int = 100):
        self.max_buffer_len: int = max_buffer_len
        self.interval_time_s: float = interval_time_ms / 1000
        self.lock: asyncio.Lock = asyncio.Lock()
        self.buffer: dict[str, str] = {}
        self.batch_queue: asyncio.Queue = asyncio.Queue()
        self.pending: dict[str, asyncio.Future] = {}

    async def add_request(self, uuid: str, message: str):
        async with self.lock:
            self.pending[uuid] = asyncio.get_event_loop().create_future()
            self.buffer[uuid] = message

            if len(self.buffer) == self.max_buffer_len:
                self.batch_queue.put_nowait(dict(self.buffer))
                self.buffer.clear()
        
        return self.pending[uuid]
    
    async def inference_worker(self):
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

            # Временная заглушка инференса
            await asyncio.sleep(2)

            # Батч остался без обработки из-за временной заглушки
            responses = batch
            for uuid, response in responses.items():
                future = self.pending.pop(uuid, None)
                if future and not future.done():
                    future.set_result(response)

inference_manager = InferenceManager()

async def lifespan(app: FastAPI):
    asyncio.create_task(inference_manager.inference_worker())
    yield
    
app = FastAPI(lifespan=lifespan)

@app.websocket("/batching")
async def websocket_endpoint(websocket: WebSocket):
    await conn_manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_json()
            req = ChatRequest(**data)
            conn_manager.add_uuid(websocket, req.uuid)
            fut = await inference_manager.add_request(req.uuid, req.message)

            async def handle_request(uuid: str, fut: asyncio.Future):
                result = await fut
                response = ChatResponse(uuid=uuid, response=result)
                await conn_manager.send_response(response)

            asyncio.create_task(handle_request(req.uuid, fut))

    except Exception as e:
        conn_manager.disconnect(websocket)
        print(f"Disconnected: {e}")
