from uuid import uuid4
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from . import summarizer

from typing import List, TypedDict, Optional
import os
import httpx

from langchain_core.messages import HumanMessage, AIMessage

# chat_langgraph에서 필요한 함수 및 클래스 추가 임포트
from .chat_langgraph import agent_app, get_chroma_client, get_embeddings
from langchain_community.vectorstores import Chroma

# langchain에서 필요한 클래스 추가 임포트
from .summarizer import summarize_news
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from fastapi import UploadFile, File, Form
import json
import tempfile

# 로컬 테스트
# uvicorn llm-service.main:app --host 0.0.0.0 --port 8000 --reload
# chroma run --host localhost --port 8001


app = FastAPI(title="AI Agent API")

# 환경 변수에서 API 키 로드
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
vector_db_host = os.getenv("VECTOR_DB_HOST", "localhost")
vector_db_port = os.getenv("VECTOR_DB_PORT", 8001)
spring_server_url = os.getenv("SPRING_SERVER_URL", "http://spring-app:8080")


class HistoryMessage(BaseModel):
    type: str
    content: str
class ChatRequest(BaseModel):
    session_id: int
    user_id: int
    question: str
    chat_message_id: int
    news_id: int  # <-- news_content 대신 news_id를 받음!
    company: str
    chat_history: List[HistoryMessage] = []

# ... (ChatResponse, HistoryMessage는 동일) ...
class ChatResponse(BaseModel):
    session_id: int
    chat_message_id: int
    question: str
    answer: str

@app.get("/")
def read_root():
    return {"message": "AI Agent 서버가 실행 중입니다. /docs 로 이동하여 API를 테스트하세요."}


@app.post("/chat", response_model=ChatResponse)
async def chat_with_file(
    request: str = Form(...),             
    file: Optional[UploadFile] = File(None)
):
    parsed_request = ChatRequest(**json.loads(request))
    
    temp_path = None    
    
    print(f"\n--- 🗣️  세션 ID {parsed_request.session_id}에 대한 요청 수신 (뉴스 ID: {parsed_request.news_id}) ---")
    
    collection_name = "news_vector_db" # collection_name을 상수로 정의

    try:
        chroma_client = get_chroma_client()
        
        try:
            # 컬렉션이 존재하는지 먼저 확인
            collection = chroma_client.get_collection(name=collection_name)
            results = collection.get(where={"news_id": str(parsed_request.news_id)}, limit=1)
        except Exception: 
            # get_collection에서 컬렉션이 없으면 예외 발생 (정확한 예외 타입은 chromadb 버전에 따라 다를 수 있음)
            results = {'ids': []} # 결과가 없는 것처럼 처리

        # VectorDB에 news_id가 없는 경우, Spring에서 가져와 저장
        if not results or not results.get('ids'):
            print(f"⚠️ ChromaDB에 news_id '{parsed_request.news_id}' 없음. Spring 서버에서 원문을 가져와 DB에 저장합니다.")
            
            # 1. Spring 서버에서 뉴스 원문 가져오기
            news_content = ""
            async with httpx.AsyncClient() as client:
                # TODO : backend url 수정
                api_url = f"{spring_server_url}/news/{parsed_request.news_id}/detail"
                print(f"Spring 서버에 뉴스 원문 요청: {api_url}")
                response = await client.get(api_url, timeout=10.0)
                response.raise_for_status()
                news_content = response.content.decode('utf-8') # 바이트를 문자열로 디코딩
                print("✅ 뉴스 원문 수신 완료")

            # 2. 가져온 원문을 VectorDB에 저장
            print("⏳ 가져온 뉴스 원문을 VectorDB에 저장하는 중...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            # Document 객체 생성 및 메타데이터 추가
            docs = [Document(page_content=chunk, metadata={"news_id": str(parsed_request.news_id)}) 
                    for chunk in text_splitter.split_text(news_content)]

            # ChromaDB에 저장
            Chroma.from_documents(
                documents=docs,
                embedding=get_embeddings(),
                client=chroma_client,
                collection_name=collection_name
            )
            print(f"✅ news_id '{parsed_request.news_id}'를 VectorDB에 성공적으로 저장했습니다.")
        else:
            print(f"✅ ChromaDB에서 news_id '{parsed_request.news_id}'를 확인했습니다.")

    except httpx.HTTPStatusError as e:
        print(f"🔥 Spring API 오류: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=424, detail=f"뉴스 원문(ID: {parsed_request.news_id})을 가져오는 데 실패했습니다.")
    except Exception as e:
        print(f"🔥 처리 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")
    

    # --- LangGraph 에이전트 실행 ---

    try:
        # ====== 파일 처리: 임시 저장 ======
        if file:
            suffix = os.path.splitext(file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                temp_path = tmp.name
            print(f"📂 파일을 '{temp_path}'에 임시 저장 완료")
        else:
            print("📂 첨부 파일 없음")

        # ========== LangGraph inputs 준비 ==========

        lc_chat_history = [
            HumanMessage(content=msg.content) if msg.type.lower() == 'human'
            else AIMessage(content=msg.content)
            for msg in parsed_request.chat_history
        ]

        # LangGraph에 전달할 입력값 구성 (GraphState에 맞게)
        inputs = {
            "session_id": parsed_request.session_id,
            "user_id": parsed_request.user_id,
            "question": parsed_request.question,
            "news_id": parsed_request.news_id,
            "company": parsed_request.company,
            "chat_history": lc_chat_history
        }
        if temp_path:
            inputs["file_path"] = temp_path
        else:
            inputs["file_path"] = None
        

        final_state = await agent_app.ainvoke(inputs)
        # 최종 답변은 'answer' 또는 'feedback' 키에 담겨 반환됨
        ai_answer = final_state.get("answer") or final_state.get("feedback", "오류: 답변을 생성하지 못했습니다.")
        
        print(f"✅ LangGraph 처리 완료")

        # print(f"--- ✅ 세션 ID {parsed_request.session_id}에 대한 응답 완료 ---")
        
        return ChatResponse(
            session_id=parsed_request.session_id,
            chat_message_id=parsed_request.chat_message_id,
            question=parsed_request.question,
            answer=ai_answer
        )
        
    except Exception as e:
        print(f"🔥 LangGraph 실행 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=f"AI 답변 생성 중 오류 발생: {e}")
    
    finally:
        # ====== 임시 파일 삭제 ======
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"🧹 임시 파일 '{temp_path}' 삭제 완료")


# =========================== summarizer POST 요청 처리 ===========================
class SummarizeRequest(BaseModel):
    id: int
    title: str
    content: str
    company_name: str
    
class SummarizeResponse(BaseModel):
    summary: str
    error: bool
    error_content: str

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(news: SummarizeRequest):

    print(f"\n--- 🗣️  뉴스 ID: {news.id}에 대한 요약 요청 수신  ---")
    
    collection_name = "news_vector_db" # collection_name을 상수로 정의

    try:
        chroma_client = get_chroma_client()
        
        try:
            # 컬렉션이 존재하는지 먼저 확인
            collection = chroma_client.get_collection(name=collection_name)
            results = collection.get(where={"id": str(news.id)}, limit=1)
        except Exception: 
            # get_collection에서 컬렉션이 없으면 예외 발생 (정확한 예외 타입은 chromadb 버전에 따라 다를 수 있음)
            results = {'ids': []} # 결과가 없는 것처럼 처리

        # VectorDB에 news_id가 없는 경우, Spring에서 가져와 저장
        if not results or not results.get('ids'):
            print(f"⚠️ ChromaDB에 news_id '{news.id}' 없음. Spring 서버에서 원문을 가져와 DB에 저장합니다.")
            
            # 1. Spring 서버에서 뉴스 원문 가져오기
            news_content = ""
            async with httpx.AsyncClient() as client:
                # TODO : backend url 수정
                api_url = f"{spring_server_url}/news/{news.id}/detail"
                print(f"Spring 서버에 뉴스 원문 요청: {api_url}")
                response = await client.get(api_url, timeout=10.0)
                response.raise_for_status()
                news_content = response.content.decode('utf-8') # 바이트를 문자열로 디코딩
                print("✅ 뉴스 원문 수신 완료")

            # 2. 가져온 원문을 VectorDB에 저장
            print("⏳ 가져온 뉴스 원문을 VectorDB에 저장하는 중...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            
            # Document 객체 생성 및 메타데이터 추가
            docs = [Document(page_content=chunk, metadata={"news_id": str(news.id)}) 
                    for chunk in text_splitter.split_text(news_content)]

            # ChromaDB에 저장
            Chroma.from_documents(
                documents=docs,
                embedding=get_embeddings(),
                client=chroma_client,
                collection_name=collection_name
            )
            print(f"✅ news_id '{news.id}'를 VectorDB에 성공적으로 저장했습니다.")
        else:
            print(f"✅ ChromaDB에서 news_id '{news.id}'를 확인했습니다.")

    except httpx.HTTPStatusError as e:
        print(f"🔥 Spring API 오류: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=424, detail=f"뉴스 원문(ID: {news.id})을 가져오는 데 실패했습니다.")
    except Exception as e:
        print(f"🔥 처리 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")
    
    # LangChain에 전달할 입력값 구성  
    summary_text = summarize_news({
        # "id": news.id,
        "title": news.title,
        "content": news.content
    })
    
    print(f"✅ LangChain 처리 완료")

    return SummarizeResponse(
        summary=summary_text,
        error=False,
        error_content=""
    )

# =========================== summarizer POST 요청 처리 ===========================
    