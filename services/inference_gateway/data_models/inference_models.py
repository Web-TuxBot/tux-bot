from pydantic import BaseModel
from uuid import UUID

class LLMRequest(BaseModel):
    requests: list[str]
    model_name: str


class LLMResponse(BaseModel):
    responses: list[str]
    created_at: str