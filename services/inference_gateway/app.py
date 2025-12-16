from fastapi import FastAPI
import asyncio
from .routers.inference_ws import inference_ws_router
from .managers.ws_connection import ClientManager, ServiceConnectionManager
from .managers.inference_manager import InferenceManager
from .logger import logger
from .settings import settings


async def lifespan(app: FastAPI):
    app.state.inferences = {}
    try:
        for inference_service in app.state.settings.inference_services:
            await app.state.service_conn_manager.connect(f"inference/{inference_service.model_name}/generate", inference_service.port, inference_service.host)
            inference = asyncio.create_task(
                app.state.inference_managers[inference_service.model_name].inference(app.state.service_conn_manager)
            )
            app.state.inferences[inference_service.model_name] = inference
        yield

    except Exception as e:
        raise RuntimeError(f"Не удалось установить WebSocket-соединение с инференс-серивсом: {e}")

    finally:
        logger.info("Завершение работы Inference Gateway...")
        for inference in list(app.state.inferences.values()):
            if inference and not inference.done():
                inference.cancel()
                try:
                    await inference
                except asyncio.CancelledError:
                    pass

        await app.state.client_manager.stop()
        await app.state.service_conn_manager.stop()

        logger.info("Inference Gateway завершил свою работу")
    

def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(inference_ws_router)
    app.state.settings = settings

    app.state.client_manager = ClientManager(time_ping_s=app.state.settings.client_time_ping_s,
                                                            pong_timeout_s=app.state.settings.client_time_pong_s)
    app.state.service_conn_manager = ServiceConnectionManager(max_delay=60)

    app.state.inference_managers = {}
    for inference_service in app.state.settings.inference_services:
        inference_manager = InferenceManager(inference_endpoint=f"inference/{inference_service.model_name}/generate",
                                            max_buffer_len=inference_service.max_buffer_len,
                                            interval_time_ms=inference_service.interval_time_ms)
        app.state.inference_managers[inference_service.model_name] = inference_manager
    
    return app