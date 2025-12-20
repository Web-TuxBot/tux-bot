from fastapi import FastAPI
from .logger import logger
from .managers.llm_manager import LLMManager
from .managers.client_manager import ClientManager
from .settings import settings
from .routers.inference_ws import inference_ws_router


async def lifespan(app: FastAPI):
    logger.info(f"Инференс-сервис с моделью {settings.model_name} запущен")
    yield
    logger.info(f"Инференс-сервис с моделью {settings.model_name} завершает свою работу...")
    app.state.client_manager.stop()
    app.state.llm_manager.stop()
    logger.info(f"Инференс-сервис с моделью {settings.model_name} завершил свою работу")


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(inference_ws_router)
    app.state.client_manager = ClientManager(max_connection=settings.max_connection)
    app.state.llm_manager = LLMManager(max_len_queue=settings.max_len_queue, model_name=settings.model_name)

    return app
