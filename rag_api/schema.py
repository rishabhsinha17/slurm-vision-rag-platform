from pydantic import BaseModel

class QueryRequest(BaseModel):
    image_url: str
    top_k: int = 5

class QueryResponse(BaseModel):
    matches: list[str]