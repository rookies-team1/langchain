from fastapi import FastAPI
from pydantic import BaseModel
from summarizer import summarize_news
from chatbot_re import run_langgraph_flow
from langGraph_p1 import run_chat_graph
from chat_langgraph import main
from typing import List, TypedDict, Optional
import os


app = FastAPI(title="AI Agent API")

# 환경 변수에서 API 키 로드
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
ollama_base_url = os.getenv("OLLAMA_BASE_UR")

# 채팅 요청 클래스
class ChatRequest(BaseModel):
    session_id: int     # 세션 id
    user_input: Optional[str] = None    # 유저 입력
    news_content: str   # 뉴스 원문
    company: str    # 회사 이름
    chat_history: List[dict] # 이전 대화기록

'''
chat_history {
    "type" : "human" or "ai", # human = 질문, ai = 답변
    "content" : 내용
}
'''

# 채팅 응답 클래스
class ChatResponse(BaseModel):
    session_id: int     # 세션 id
    question: str   # 질문
    answer: str     # 답변
    # timestamp (필요하면)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):

    # LangGraph를 사용하여 답변 생성
    ai_response = run_chat_graph(
        user_input=request.user_input,
        news_content=request.news_content,
        chat_history=request.chat_history
    )

    return ChatResponse(question=request.user_input, answer=ai_response, session_id=request.session_id)


# =========================== summarizer POST 요청 처리 ===========================
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
    
# =========================== summarizer POST 요청 처리 ===========================
    
    

# LLM 기반 멀티 기능 처리 요청
class ChatbotRequest(BaseModel):
    user_question: str
    resume_path: str = None  # optional
    news_content: str = None
    news_summary_path: str

@app.post("/chatbot")
def chatbot_router(request: ChatbotRequest):
    try:
        result = main(
            user_question=request.user_question,
            resume_path=request.resume_path,
            news_content=request.news_content,
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