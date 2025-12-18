from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime
from ..data_models.inference_models import LLMRequest, LLMResponse
from ..settings import settings
from ..logger import logger


inference_ws_router = APIRouter()


@inference_ws_router.websocket(f"/inference/{settings.model_name}/generate")
async def websocket_endpoint(ws: WebSocket):
    app = ws.app
    try:
        await app.state.client_manager.connect(ws)

    except Exception as e:
        return

    try:
        while True:
            try:
                data = await ws.receive_json()
            except RuntimeError as e:
                logger.error(f"Возникла ошибка при попытке принять сообщение от клиента: {e}")
                raise WebSocketDisconnect

            batch = LLMRequest(**data).requests

            fut = await app.state.llm_manager.add_batch(batch, ws)
            responses = await fut

            created_at = datetime.now().isoformat()
            responses = LLMResponse(responses=responses, created_at=created_at)

            await app.state.client_manager.send_response(responses, ws)

    except WebSocketDisconnect:
        if fut and not fut.done():
            fut.cancel()
        await app.state.client_manager.disconnect(ws)
