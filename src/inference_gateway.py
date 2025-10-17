from fastapi import FastAPI, WebSocket
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = ""

    def add_uuid(self, websocket: WebSocket, uuid: str):
        self.active_connections[websocket] = uuid
    
    def disconnect(self, websocket: WebSocket):
        del self.active_connections[websocket]
    
    async def send_response(self, websocket: WebSocket, response: dict):
        try:
            await websocket.send_json(response)
        except Exception:
            self.disconnect(websocket)

conn_manager = ConnectionManager()

#TODO(mortiferr): Убрать временную заглушку инференса и реализовать подключение к инференс-сервису
class InferenceManager:
    def __init__(self, max_buffer_len: int = 10, interval_time_ms: int = 100):
        self.max_buffer_len: int = max_buffer_len
        self.interval_time_s: float = interval_time_ms / 1000
        self.lock: asyncio.Lock = asyncio.Lock()
        self.buffer: dict = {}
        self.batch_queue: asyncio.Queue = asyncio.Queue()
        self.pending: dict[WebSocket, asyncio.Future] = {}

    async def add_request(self, websocket: WebSocket, prompt: str):
        async with self.lock:
            self.pending[websocket] = asyncio.get_event_loop().create_future()
            self.buffer[websocket] = prompt

            if len(self.buffer) == self.max_buffer_len:
                self.batch_queue.put_nowait(dict(self.buffer))
                self.buffer.clear()
        
        return self.pending[websocket]
    
    async def inference_worker(self):
        while True:
            await asyncio.sleep(self.interval_time_s)
            async with self.lock:
                batch = []
                if self.batch_queue.empty():
                    if not self.buffer:
                        continue
                    batch = self.buffer.copy()
                    self.buffer.clear()
                else:
                    batch = await self.batch_queue.get()

            # Временная заглушка инференса
            await asyncio.sleep(2)

            # Батч остался без обработки из-за временной заглушки
            responses = batch
            for ws, response in responses.items():
                future = self.pending.pop(ws, None)
                future.set_result(response)


inference_manager = InferenceManager(interval_time_ms=2000)

async def lifespan(app: FastAPI):
    asyncio.create_task(inference_manager.inference_worker())
    yield
    
app = FastAPI(lifespan=lifespan)

@app.websocket("/batching")
async def websocket_endpoint(websocket: WebSocket):
    await conn_manager.connect(websocket)
    try:
        while True:
            request = await websocket.receive_json()
            conn_manager.add_uuid(websocket, request["uuid"])
            fut = await inference_manager.add_request(websocket, request["prompt"])
            response = await fut
            await conn_manager.send_response(websocket, {"uuid": request["uuid"], "response": response})
    except Exception as e:
        conn_manager.disconnect(websocket)
        print(f"Disconnected: {e}")
