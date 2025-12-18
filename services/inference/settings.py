from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str
    max_len_queue: int
    max_connection: int


settings = Settings()