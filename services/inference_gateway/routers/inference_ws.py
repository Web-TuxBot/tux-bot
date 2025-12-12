from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
from uuid import UUID
from pydantic import ValidationError
from ..data_models.chat_models import ChatRequest, ChatResponse, ClientReconnect
from ..logger import logger


inference_ws_router = APIRouter()


@inference_ws_router.websocket("/inference/batching")
async def inference_ws(ws: WebSocket):
    app = ws.app
    handles = []
    await app.state.client_conn_manager.connect(ws)
    try:
        while True:
            data = await ws.receive_json()

            try:
                req = ChatRequest(**data)

            except ValidationError:
                req = ClientReconnect(**data)
                await app.state.client_conn_manager.reconnect(ws, tuple(req.uuids))
                continue

            await app.state.client_conn_manager.add_uuid(ws, req.uuid)

            fut = await app.state.inference_manager.add_request(req.uuid, req.message)

            async def handle_request(uuid: UUID, fut: asyncio.Future) -> None:
                try:
                    response, created_at = await fut
                    response = ChatResponse(uuid=uuid, response=response, created_at=created_at)
                    await app.state.client_conn_manager.send_response(response)

                except asyncio.CancelledError as e:
                    await app.state.inference_manager.cancel_future(uuid)
                
            handles.append(asyncio.create_task(handle_request(req.uuid, fut)))
            handles = [h for h in handles if not h.done()]
            
    except WebSocketDisconnect as e:
        event_idx = await app.state.client_conn_manager.disconnect(ws, reconnect=True)
        successfuly = await app.state.client_conn_manager.get_reconnect_status(event_idx)
        if not successfuly:
            for handle in handles:
                handle.cancel()
                try:
                    await handle
                except asyncio.CancelledError as e:
                    continue
            handles.clear()