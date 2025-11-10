from pydantic import BaseModel
from uuid import UUID

class ChatRequest(BaseModel):
    uuid: UUID
    model_name: str
    message: str


class ChatResponse(BaseModel):
    uuid: UUID
    response: str


class LLMRequest(BaseModel):
    requests: list[str]


class LLMResponse(BaseModel):
    responses: list[str]
    created_at: str

