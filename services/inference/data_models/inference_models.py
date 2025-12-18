from pydantic import BaseModel


class LLMRequest(BaseModel):
    requests: list[str]


class LLMResponse(BaseModel):
    responses: list[str]
    created_at: str