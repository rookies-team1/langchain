from fastapi import FastAPI
from pydantic import BaseModel
from summarizer import summarize_news
from chatbot_re import run_langgraph_flow
import os


app = FastAPI(title="AI Agent API")

# 환경 변수에서 API 키 로드
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
 
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
    
# LLM 기반 멀티 기능 처리 요청
class ChatbotRequest(BaseModel):
    user_question: str
    resume_path: str = None  # optional
    news_full_path: str = None
    news_summary_path: str

@app.post("/chatbot")
def chatbot_router(request: ChatbotRequest):
    try:
        result = run_langgraph_flow(
            user_question=request.user_question,
            resume_path=request.resume_path,
            news_full_path=request.news_full_path,
            news_summary_path=request.news_summary_path
        )
        return {
            "status": "success",
            "next_node": result.get("next_node"),
            "answer": result.get("answer"),
            "feedback": result.get("feedback"),
            "chat_history": result.get("chat_history"),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# uvicorn main:app --host 0.0.0.0 --port 8000 --reload