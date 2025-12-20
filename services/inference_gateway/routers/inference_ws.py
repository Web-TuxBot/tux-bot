from fastapi import APIRouter, WebSocket, WebSocketDisconnect 
import asyncio 
from uuid import UUID  
from ..data_models.chat_models import ChatRequest, ChatResponse 
from ..logger import logger 


inference_ws_router = APIRouter() 


@inference_ws_router.websocket("/inference/batching") 
async def inference_ws(ws: WebSocket): 
    app = ws.app 
    handles = []

    try: 
        await app.state.client_manager.connect(ws) 
    except (WebSocketDisconnect, RuntimeError):
        return

    try: 
        while True:
            try:
                data = await ws.receive_json() 
            except RuntimeError as e:
                logger.error(f"Возникла ошибка при попытке принять сообщение от клиента: {e}")
                raise WebSocketDisconnect

            if data.get("type") == "pong": 
                await app.state.client_manager.handle_pong(ws) 
                continue 

            req = ChatRequest(**data) 

            if req.model_name not in app.state.inference_managers:
                logger.warning(f"Запрошен запрос к неизвестной модели: {req.model_name}")
                raise WebSocketDisconnect

            if app.state.inferences[req.model_name].done():
                raise WebSocketDisconnect
                a
            
            await app.state.client_manager.add_uuid(ws, req.uuid) 
            fut = await app.state.inference_managers[req.model_name].add_request(req.uuid, req.message) 

            async def handle_request(uuid: UUID, fut: asyncio.Future, model_name: str) -> None:
                try: 
                    response, created_at = await fut 
                    response = ChatResponse(uuid=uuid, response=response, created_at=created_at) 
                    await app.state.client_manager.send_response(response) 

                except asyncio.CancelledError as e: 
                    await app.state.inference_managers[model_name].cancel_future(uuid)

            handles.append(asyncio.create_task(handle_request(req.uuid, fut, req.model_name))) 
            handles = [h for h in handles if not h.done()] 

    except WebSocketDisconnect as e: 
        await app.state.client_manager.disconnect(ws)
        for handle in handles:
            handle.cancel()
            try:
                await handle
            except asyncio.CancelledError:
                pass