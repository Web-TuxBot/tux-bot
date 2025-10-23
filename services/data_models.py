from pydantic import BaseModel

class ChatRequest(BaseModel):
    uuid: str
    model_name: str
    message: str

class ChatResponse(BaseModel):
    uuid: str
    response: str