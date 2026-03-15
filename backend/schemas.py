from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
    ticker: str
    use_fmi: bool = False
    thread_id: str | None = None
