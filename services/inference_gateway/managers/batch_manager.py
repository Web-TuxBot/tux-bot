import asyncio
from uuid import UUID
from ..logger import logger


class BatchManager:
    def __init__(self, max_buffer_len: int = 10, interval_time_ms: int = 100):
        self.max_buffer_len = max_buffer_len
        self.interval_time_s = interval_time_ms / 1000
        self.timer_task = asyncio.create_task(self._timer())
        self.lock = asyncio.Lock()
        self.buffer = {}
        self.batch_queue = asyncio.Queue()
        logger.debug(f"Инициализирован батчер запросов."
                        "Параметры:"
                        f"Длина буфера: {self.max_buffer_len}"
                        f"Время сбора батча: {self.interval_time_s} с")
    
    async def _timer(self):
        try:
            while True:
                await asyncio.sleep(self.interval_time_s)
                async with self.lock:
                    if self.buffer:
                        self.batch_queue.put_nowait(self.buffer.copy())
                        self.buffer.clear() 

        except asyncio.CancelledError:
            return

    async def stop(self):
        self.timer_task.cancel()
        await self.timer_task

    async def add_to_buffer(self, uuid: UUID, message: str):
        async with self.lock:
            self.buffer[uuid] = message

            if len(self.buffer) >= self.max_buffer_len:
                self.batch_queue.put_nowait(self.buffer.copy())
                self.buffer.clear()

    async def get_batch(self) -> dict[str, str]:
        batch = await self.batch_queue.get()
        return batch