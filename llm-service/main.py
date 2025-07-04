from fastapi import FastAPI
from pydantic import BaseModel
from summarizer import summarize_news

app = FastAPI(title="AI Agent API")

@app.get("/")
def read_root():
    return {"message": "LLM Service is running."}

# summarizer POST 요청 처리

class News(BaseModel):
    title: str
    content: str

@app.post("/summarize")
async def summarize(news: News):
    try:
        summary = summarize_news(news.dict())
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}