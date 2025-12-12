from fastapi import FastAPI
import asyncio
from .routers.inference_ws import inference_ws_router
from .managers.ws_connection import ClientConnectionManager, ServiceConnectionManager
from .managers.inference_manager import InferenceManager
from .logger import logger
from .settings import settings


async def lifespan(app: FastAPI):
    inference = None
    try:
        await app.state.service_conn_manager.connect("inference/generate", settings.inference.port, settings.inference.host)
        inference = asyncio.create_task(
            app.state.inference_manager.inference(app.state.service_conn_manager)
        )

        yield

    except Exception as e:
        raise RuntimeError(f"Не удалось установить WebSocket-соединение с инференс-серивсом: {e}")

    finally:
        logger.info("Завершение работы Inference Gateway...")
        logger.info("Завершение работы InferenceManager...")
        if inference:
            inference.cancel()
            try:
                await inference
            except asyncio.CancelledError:
                pass

        await app.state.service_conn_manager.stop()
        await app.state.client_conn_manager.stop()

        logger.info("InferenceManager завершил свою работу")
        logger.info("Inference Gateway завершил свою работу")
    

def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.include_router(inference_ws_router)
    app.state.settings = settings
    app.state.client_conn_manager = ClientConnectionManager()
    app.state.service_conn_manager = ServiceConnectionManager(max_delay=60)
    app.state.inference_manager = InferenceManager(model_name="qwen2-5_instruct", 
                                                   inference_endpoint="inference/generate",
                                                   max_buffer_len=app.state.settings.batcher.max_buffer_len,
                                                   interval_time_ms=app.state.settings.batcher.interval_time_ms)
    
    return app