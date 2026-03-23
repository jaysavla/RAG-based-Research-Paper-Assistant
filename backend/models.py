from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class AskRequest(BaseModel):
    query: str
    top_k: int = 8


class EvalGenRequest(BaseModel):
    samples_per_doc: int = 5


class EvalRunRequest(BaseModel):
    k: int = 5
