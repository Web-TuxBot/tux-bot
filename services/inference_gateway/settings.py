from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceService(BaseModel):
    model_name: str
    host: str
    port: int

    max_buffer_len: int = Field(default=50)
    interval_time_ms: int = Field(default=100)


class Settings(BaseSettings):
    #model_config = SettingsConfigDict(env_nested_max_split=1, env_nested_delimiter="_")
    inference_services: list[InferenceService]
    client_time_ping_s: int = Field(default=30)
    client_time_pong_s: int = Field(default=25)


settings = Settings()
