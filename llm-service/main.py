from fastapi import FastAPI
from pydantic import BaseModel
from summarizer import summarize_news
from resume_analyzer import process_resume_with_news

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
    
# 이력서 분석 및 피드백 요청 처리
class FeedbackRequest(BaseModel):
    resume_path: str
    news_path: str

@app.post("/feedback")
def analyze_resume(request: FeedbackRequest):
    try:
        result = process_resume_with_news(
            resume_path=request.resume_path,
            news_path=request.news_path
        )
        return {"status": "success", "feedback": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}