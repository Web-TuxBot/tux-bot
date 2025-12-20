import asyncio
from asyncio import Future
from fastapi import WebSocket
from ..models.llm_model import LLMModel


class LLMManager:
    def __init__(self, model_name: str, max_len_queue: int):
        self.batches = asyncio.Queue(maxsize=max_len_queue)
        self.pending = {}
        self.model = LLMModel(model_name=model_name)
        self.inference = asyncio.create_task(self._inference_loop())
    
    async def stop(self):
        self.inference.cancel()
        try:
            await self.inference
        except asyncio.CancelledError as e:
            pass

        for ws in list(self.pending.keys()):
            self.cancel_future(ws)
    
    def cancel_future(self, ws: WebSocket):
        fut = self.pending.pop(ws, None)
        if fut and not fut.done():
            fut.cancel()

    async def _inference_loop(self) -> None:
        try:
            while True:
                ws, batch = await self.batches.get()
                responses = await asyncio.to_thread(self.model.get_response, batch)
                if ws in self.pending:
                    fut = self.pending.pop(ws, None)
                    if fut and not fut.done():
                        fut.set_result(responses)

        except asyncio.CancelledError as e:
            pass

    async def add_batch(self, batch: list[str], ws: WebSocket) -> Future:
        self.pending[ws] = asyncio.get_event_loop().create_future()
        await self.batches.put((ws, batch))
        return self.pending[ws]
