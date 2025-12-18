from ..data_models.inference_models import LLMRequest, LLMResponse
from .service_manager import ServiceConnectionManager
from .batch_manager import BatchManager
import asyncio
from asyncio import Future
from websockets import ConnectionClosed
from uuid import UUID
from ..logger import logger


class InferenceManager:
    def __init__(self,
                 inference_endpoint: str,
                 max_buffer_len: int = 10, 
                 interval_time_ms: int = 100):
        self.inference_endpoint = inference_endpoint
        self.batcher = BatchManager(max_buffer_len=max_buffer_len, interval_time_ms=interval_time_ms)
        self.pending = {}
        logger.info(f"Инициализирован менеджер инференса: inference endpoint - {self.inference_endpoint}")
        
    async def inference(self, service_conn_manager: ServiceConnectionManager) -> None:
        try:
            while True:
                batch = await self.batcher.get_batch()
                requests, uuids = [], []

                for uuid, request in batch.items():
                    requests.append(request)
                    uuids.append(uuid)
                requests = LLMRequest(requests=requests)
                
                try:
                    await service_conn_manager.safe_send(self.inference_endpoint, requests.model_dump(mode="json"))
                    responses = await service_conn_manager.safe_recv(self.inference_endpoint, requests.model_dump(mode="json"))
                    responses = LLMResponse(**responses)

                except ConnectionClosed as e:
                    logger.error("Не удалось отправить/получить сообщения с инференс-сервиса: Инференс-сервис закрыт")
                    for uuid in uuids:
                        self.cancel_future(uuid)
                    await self.batcher.stop()
                    return

                except Exception as e:
                    logger.critical(f"Критическая ошибка при попытке отправить/получить сообщения с инференс-сервиса: {e}")
                    for uuid in uuids:
                        self.cancel_future(uuid)
                    await self.batcher.stop()
                    return
                
                created_at = responses.created_at
                
                responses_dict = {}
                for response, uuid in zip(responses.responses, uuids):
                    responses_dict[uuid] = response 

                self.send_response(responses_dict, created_at)

        except asyncio.CancelledError as e:
            await self.batcher.stop()
            raise e

    async def add_request(self, uuid: UUID, message: str) -> Future:
        self.pending[uuid] = asyncio.get_event_loop().create_future()
        await self.batcher.add_to_buffer(uuid, message)
        return self.pending[uuid]
    
    async def cancel_future(self, uuid: UUID):
        if uuid in self.pending:
            fut = self.pending.pop(uuid)
            if fut and not fut.done():
                fut.cancel()
                try:
                    await fut
                except asyncio.CancelledError:
                    pass
    
    def send_response(self, responses: dict[str, str], created_at: str) -> None:
        for uuid, response in responses.items():
            if uuid in self.pending:
                fut = self.pending.pop(uuid) 
                if fut and not fut.done():
                    fut.set_result((response, created_at))
            else:
                continue