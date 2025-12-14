from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class BatcherSettings(BaseModel):
    max_buffer_len: int
    interval_time_ms: int


class InferenceSettings(BaseModel):
    port: int
    host: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_max_split=1, env_nested_delimiter="_")
    batcher: BatcherSettings
    inference: InferenceSettings
    client_time_ping_s: int
    client_time_pong_s: int

settings = Settings()
