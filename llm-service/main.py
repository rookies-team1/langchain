from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from summarizer import summarize_news
from chatbot_re import run_langgraph_flow
from chat_langgraph import main
from typing import List, TypedDict, Optional
import os
import httpx

from langchain_core.messages import HumanMessage, AIMessage
from chat_langgraph import agent_app

app = FastAPI(title="AI Agent API")

# 환경 변수에서 API 키 로드
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
spring_server_url = os.getenv("SPRING_SERVER_URL", "http://localhost:8080")


class HistoryMessage(BaseModel):
    type: str
    content: str
class ChatRequest(BaseModel):
    session_id: int
    user_input: str
    news_id: int  # <-- news_content 대신 news_id를 받음!
    company: str
    chat_history: List[HistoryMessage] = []

# ... (ChatResponse, HistoryMessage는 동일) ...
class ChatResponse(BaseModel):
    session_id: int
    question: str
    answer: str

@app.get("/")
def read_root():
    return {"message": "AI Agent 서버가 실행 중입니다. /docs 로 이동하여 API를 테스트하세요."}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    print(f"\n--- 🗣️  세션 ID {request.session_id}에 대한 요청 수신 (뉴스 ID: {request.news_id}) ---")
    
    news_content = ""
    # --- Spring API 호출하여 뉴스 원문 가져오기 ---
    try:
        # 비동기 HTTP 클라이언트 사용
        async with httpx.AsyncClient() as client:
            api_url = "spring 서버 url"
            print(f"Spring 서버에 뉴스 원문 요청: {api_url}")
            response = await client.get(api_url, timeout=5.0)
            
            # 응답 상태 코드 확인
            response.raise_for_status() # 2xx가 아니면 예외 발생
            
            news_content = response.text
            print("✅ 뉴스 원문 수신 완료")

    except httpx.HTTPStatusError as e:
        print(f"🔥 Spring API 오류: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=424, detail=f"뉴스 원문(ID: {request.news_id})을 가져오는 데 실패했습니다.")
    except Exception as e:
        print(f"🔥 뉴스 원문 조회 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail="내부 서버 간 통신 오류")
    # ----------------------------------------------

    try:
        # LangChain 메시지 객체로 변환
        lc_chat_history = [
            HumanMessage(content=msg.content) if msg.type.lower() == 'human'
            else AIMessage(content=msg.content)
            for msg in request.chat_history
        ]

        # agent.py에 정의된 LangGraph 에이전트 실행
        inputs = {
            "user_input": request.user_input,
            "news_content": news_content, # <-- 조회해 온 원문을 전달
            "company": request.company,
            "chat_history": lc_chat_history,
        }

        final_state = await agent_app.ainvoke(inputs)
        ai_answer = final_state.get("answer", "오류: 답변을 생성하지 못했습니다.")
        
        print(f"--- ✅ 세션 ID {request.session_id}에 대한 응답 완료 ---")
        
        return ChatResponse(
            session_id=request.session_id,
            question=request.user_input,
            answer=ai_answer
        )
        
    except Exception as e:
        print(f"🔥 LangGraph 실행 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=f"AI 답변 생성 중 오류 발생: {e}")


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