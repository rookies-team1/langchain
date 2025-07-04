from fastapi import FastAPI
import os
from pydantic import BaseModel
from summarizer import summarize_news

app = FastAPI(title="AI Agent API")

# 환경 변수에서 API 키 로드
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
 
@app.get("/")
def read_root():
    if google_api_key:
        return {"message": "LLM Service is running.", "api_key_status": "loaded", "key" : google_api_key[:2]}
    else:
        return {"message": "LLM Service is running.", "api_key_status": "not loaded"}
    

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