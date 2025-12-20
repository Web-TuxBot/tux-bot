from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceService(BaseModel):
    model_name: str
    host: str
    port: int

    max_buffer_len: int = Field(default=50)
    interval_time_ms: int = Field(default=100)


class Settings(BaseSettings):
    inference_services: list[InferenceService]
    client_ping_interval: int = Field(default=30)
    client_ping_timeout: int = Field(default=25)
    service_ping_interval: int = Field(default=30)
    service_ping_timeout: int = Field(default=25)
    service_max_delay: int = Field(default=60)


settings = Settings()
