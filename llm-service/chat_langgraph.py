# poetry run python ./llm-service/chat_langgraph_2.py

import os
import sys

from langchain_openai import ChatOpenAI

# llm-service 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from typing import TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv
from collections import defaultdict
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langsmith import Client
from langsmith import traceable
from langsmith import traceable
import json
from langchain_chroma import Chroma
from chromadb import chromadb
import re
import chromadb

# 로컬 ChromaDB 클라이언트 설정
# chroma_client = chromadb.HttpClient(host="localhost", port=8001)

# ==============================================================================
# 1. 초기화 및 설정
# ==============================================================================

llm = None
embeddings = None
tavily_tool = TavilySearch(k=3)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# LangSmith API Key 설정
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "llm-service-already")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
client = Client(api_key=LANGSMITH_API_KEY)

def get_llm():
    """LLM 인스턴스를 가져옵니다. 없으면 생성합니다."""
    global llm
    if llm is None:
        load_dotenv()
        # llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.5-pro",
        #     temperature=0.7,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2,
        # )
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url="https://api.groq.com/openai/v1",  # Groq API 엔드포인트
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.7
        )
    return llm

def get_embeddings():
    """임베딩 모델 인스턴스를 가져옵니다. 없으면 생성합니다."""
    global embeddings
    if embeddings is None:
        load_dotenv()
        try:
            # ChromaDB와 같은 영구적인 저장소를 사용할 것이므로, 일관된 임베딩 모델 사용이 중요
            embeddings = OllamaEmbeddings(
                model="bge-m3:latest", 
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
        except Exception as e:
            print(f"임베딩 모델 로드 실패: {e}")
            raise
    return embeddings


def get_chroma_client():
    """ChromaDB 클라이언트 인스턴스를 가져옵니다. 없으면 생성합니다."""
    global chroma_client
    if chroma_client is None:
        load_dotenv()
        # 환경 변수 또는 기본값으로 ChromaDB에 연결
        CHROMA_HOST = os.getenv("VECTOR_DB_HOST", "localhost")
        CHROMA_PORT = int(os.getenv("VECTOR_DB_PORT", "8001")) # ChromaDB 기본 포트는 8000이나, docker-compose 예시에서 8001로 설정
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return chroma_client

# ==============================================================================
# 2. Graph State 정의
# ==============================================================================

class GraphState(TypedDict):
    # 입력 값
    session_id: int
    user_id: int
    question: str
    news_id: int  # 뉴스 식별자
    file_path: Optional[str]
    company: Optional[str]  # 기업명 (Tavily 검색에 사용)
    chat_history: List[BaseMessage]
    # 그래프 내부에서 관리되는 값
    input_type: str  # 'qa' or 'feedback'
    # question: str  # 재구성된 질문
    
    relevant_chunks: List[str]
    # QA 경로 관련 상태
    retriever: Optional[Any] # Retriever 객체 저장 (수정: 상태에 retriever 추가)
    
    is_grounded: bool
    tavily_snippets: Optional[List[str]]  # Tavily 검색 결과 스니펫

    # Feedback 경로 관련 상태
    pages: Optional[List]
    user_file_summary: Optional[str]

    # 답변 관리
    answer: str

# ==============================================================================
# 3. LangGraph 노드 함수 정의
# ==============================================================================

def clean_llm_output(text: str) -> str:
    # LLM의 출력물에서 필요 없는 태그/마크다운/불필요한 줄바꿈 정리

    # 불필요한 역슬래시 2개 이상 → 1개로 치환
    text = re.sub(r'\\\\+', r'\\', text)
    # \_ → _ 로 변환 (필요시)
    text = text.replace('\\_', '_')
    # <think>...</think> 블록 제거
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 출력 시작/끝에 markdown block이 남는 경우 제거
    text = text.strip()
    # markdown block 안에만 남아있는 경우 잘라내기
    # 예: ```markdown ... ``` 구조 제거
    text = re.sub(r"```(?:markdown)?\s*(.*?)```", r"\1", text, flags=re.DOTALL | re.IGNORECASE)
    # 연속되는 3줄 이상 줄바꿈은 2줄로 축소
    text = re.sub(r"\n{3,}", "\n\n", text)
     # ```json ... ``` 등 코드블록 제거
    text = re.sub(r"```json?(.*?)```", r"\1", text, flags=re.DOTALL|re.IGNORECASE)
    # 혹은 ``` ... ``` 전체 제거
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # 필요 없는 선두/후미 공백 제거
    return text.strip()


def clean_pdf_text(text: str) -> str:
    # 1) 여러 줄바꿈을 하나로 줄이기
    text = re.sub(r'\n+', '\n', text)
    
    # 2) 단어 사이 줄바꿈(\n) -> 공백으로 대체 (단, 문장 끝 \n은 살릴 수 있음)
    # 예: '임베디드\n시스템' → '임베디드 시스템'
    text = re.sub(r'(?<=\S)\n(?=\S)', ' ', text)
    
    # 3) 다중 공백을 한 칸 공백으로 축소
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 4) 문장 부호 뒤에는 줄바꿈 살리고 나머지는 띄어쓰기
    # (필요 시 커스텀)
    
    return text.strip()
import json

@traceable(run_type="chain", name="Simple_Chain")
def retrieve_from_chroma_node(state: GraphState) -> GraphState:
    """ChromaDB에서 news_id를 필터링하여 관련 뉴스 청크를 검색하여 state['relevant_chunks']에 할당."""

    print(f"--- 1. ChromaDB 뉴스 검색 시작 (news_id={state['news_id']}) ---")
    
    embeddings = get_embeddings()
    chroma_client = get_chroma_client()
    
    vectorstore = Chroma(
        client=chroma_client,
        collection_name="news_vector_db",
        embedding_function=embeddings,
    )
    
    news_id_filter = str(state['news_id'])
    
    retriever = vectorstore.as_retriever(
        search_kwargs={
            'filter': {'news_id': news_id_filter},
            'k': 5  # 필요시 개수 조정
        }
    )
    
    question = state['question']
    
    try:
        if hasattr(retriever, 'invoke'):
            documents = retriever.invoke(question)
        else:
            documents = retriever.invoke(question)

    except Exception as e:
        print(f"❌ Chroma 검색 중 오류 발생: {e}")
        state['relevant_chunks'] = []
        return state

    if not documents:
        print(f"⚠️ news_id '{news_id_filter}'에 해당하는 관련 청크를 찾을 수 없습니다.")
        state['relevant_chunks'] = []
        return state

    extracted_chunks = []
    for idx, doc in enumerate(documents):
        raw_content = doc.page_content.strip()
        if not raw_content:
            print(f"⚠️ 빈 콘텐츠 발견 (index={idx}), 스킵합니다.")
            continue

        try:
            # JSON 형식이라 판단되면 파싱 시도
            if raw_content.startswith('{') and raw_content.endswith('}'):
                parsed_json = json.loads(raw_content)
                # data.contents 경로에 실제 텍스트가 있을 경우만 추출
                extracted_text = parsed_json.get("data", {}).get("contents", raw_content)
            else:
                extracted_text = raw_content
        except Exception as e:
            print(f"⚠️ JSON 파싱 실패, 전체 raw_content 사용 (index={idx}): {e}")
            extracted_text = raw_content

        extracted_chunks.append(extracted_text)

    state['relevant_chunks'] = extracted_chunks

    print(f"✅ 검색 완료: '{question[:30]}...' 에 대해 {len(extracted_chunks)}개의 관련 청크를 할당했습니다.")
    return state


# --- 라우팅 노드 ---
@traceable(run_type="chain", name="Simple_Chain")
def route_request_node(state: GraphState) -> dict:
    """사용자 질문을 분석하여 다음 단계를 결정하는 라우터"""
    print("--- 2. 요청 라우팅 ---")
    llm = get_llm()
    route_prompt = ChatPromptTemplate.from_template(
        """사용자 질문 '{question}'은 다음 중 어떤 유형에 가장 가깝습니까?
        - 뉴스 기사에 대한 질문: 'qa'
        - 첨부된 문서(이력서/포트폴리오)에 대한 피드백 요청: 'feedback'
        답변은 반드시 'qa' 또는 'feedback' 단어 하나만 포함해야 합니다."""
    )
    routing_chain = route_prompt | llm | StrOutputParser()
    result = routing_chain.invoke({"question": state["question"]})
    
    cleaned_result = clean_llm_output(result).lower()
    print(f"✅ LLM 분기 판단 결과: {cleaned_result}")

    if "feedback" in cleaned_result:
        return {"input_type": "feedback"}
    elif "qa" in cleaned_result:
        return {"input_type": "qa"}
    else:
        # 기본값으로 QA 설정 또는 에러 처리
        print("⚠️ 라우팅 실패, 기본값 'qa'로 설정")
        return {"input_type": "qa"}
    
    
# --- 뉴스 Q&A 경로 ---
@traceable(run_type="chain", name="Simple_Chain")
def get_tavily_snippets(state: GraphState):
    """
    Tavily를 사용해 기업명 + 사용자 질문 기반의 최신 웹 스니펫을 검색하여 반환.
    """
    print("--- 3a. Taviliy search ---")
    try:
        question = state.get('question')
        company_name = state.get('company')

        # 검색 쿼리 구성
        if company_name and question:
            search_query = f"{company_name} 관련 {question}"
        else:
            raise ValueError("검색할 질문과 기업명이 모두 비어 있습니다.")
        
        print(f"🔍 Tavily 검색 쿼리: {search_query}")

        # Tavily 검색
        results = tavily_tool.invoke(search_query)

        snippets = []
        for item in results.get('results', []):
            # 검색된 snippet과 출처 URL 함께 구성
            snippet = f"{item.get('content', '').strip()}\n출처: {item.get('url', '').strip()}"
            if snippet.strip():
                snippets.append(snippet)

        print(f"✅ Tavily: {len(snippets)}개 스니펫 검색 완료")
        state["tavily_snippets"] = snippets

        return state

    except Exception as e:
        print(f"⚠️ Tavily 검색 실패: {e}")
        return state
    

@traceable(run_type="chain", name="Simple_Chain")
def generate_answer_node(state: GraphState):
    print("--- 4a. 답변 생성 ---")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """다음 [뉴스 기사 내용]과 [웹 검색 스니펫]을 참고하여 [질문]에 대해 한국어로 명확하고 간결하게 답변하세요.
        [뉴스 기사 내용]: {context}
        [기업 관련 검색 스니펫]: {web_snippets}
        [질문]: {question}"""
    )
    rag_chain = prompt | llm | StrOutputParser()
    
    # Tavily snippet 추가
    tavily_snippets = state.get('tavily_snippets', [])
    tavily_context = "\n\n".join(tavily_snippets) if tavily_snippets else "검색된 웹 스니펫 없음."


    if not state['relevant_chunks']:
        answer = "관련 정보를 찾을 수 없습니다. 뉴스 ID를 확인해주세요."
    else:
        answer = rag_chain.invoke({
            "context": "\n---\n".join(state['relevant_chunks']),
            "web_snippets": tavily_context,
            "question": state['question']
        })
    state['answer'] = clean_llm_output(answer)
    return state

@traceable(run_type="chain", name="Simple_Chain")
def grade_answer_node(state: GraphState):
    print("--- 5a. 답변 검증 ---")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """[뉴스 기사 내용]을 볼 때, [생성된 답변]이 [질문]에 대해 사실에 근거하는지 평가해주세요.
        근거했다면 'yes', 아니면 'no'라고만 답해주세요.
        [뉴스 기사 내용]: {context}
        [질문]: {question}
        [생성된 답변]: {answer}"""
    )
    grading_chain = prompt | llm | StrOutputParser()
    grade = grading_chain.invoke({
        "context": "\n---\n".join(state['relevant_chunks']),
        "question": state['question'],
        "answer": state['answer']
    })
    
    if "yes" in grade.lower():
        state['is_grounded'] = True
        print("✅ 검증 결과: 통과")
    else:
        state['is_grounded'] = False
        print("❌ 검증 결과: 실패")
    return state


# --- 문서 피드백 경로 ---
@traceable(run_type="chain", name="Simple_Chain")
def load_and_summarize_resume_node(state: GraphState):

    print("--- 3b. 이력서 로드 및 요약 ---")

    file_path = state['file_path']

    if not file_path or not os.path.exists(file_path):
        print("📂 파일 경로가 없으므로 ChromaDB에서 기존 요약을 조회합니다.")
        chroma_client = get_chroma_client()
        collection = chroma_client.get_or_create_collection(name="user_resume_db")

        query_id = f"{state['session_id']}_{state['user_id']}"
        results = collection.get(ids=[query_id])
        
        if results and results.get('documents'):
            state['user_file_summary'] = results['documents'][0]
            print(f"✅ ChromaDB에서 이력서 요약 복원 완료: {state['user_file_summary']}...")
            return state
        else:
            raise ValueError(
                f"파일 경로가 없고 ChromaDB에도 요약된 파일이 없습니다. "
                f"세션 {state.get('session_id')}, 사용자 {state.get('user_id')}. "
                "피드백을 원하는 파일을 첨부해 주세요."
            )
    
    else:
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
        else: # txt
            loader = TextLoader(file_path, encoding='utf-8')
            pages = loader.load_and_split()
        
        full_text = " ".join([page.page_content for page in pages])
        full_text = clean_pdf_text(full_text)

        # (비용/시간 개선) 전체 텍스트를 한번에 요약하도록 변경
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(
            "다음 이력서/포트폴리오의 핵심 역량과 프로젝트 경험을 3~5 문장으로 요약해줘.\n\n{text}")
        summarization_chain = prompt | llm | StrOutputParser()
        summary = summarization_chain.invoke({"text": full_text})
        
        state['user_file_summary'] = f"{clean_llm_output(summary)}"
        print(f"✅ 이력서 요약 완료: {state['user_file_summary']}...")  # 요약의 일부만 출력
        print("✅ 이력서 요약 완료")

        # ---- store uploaded file summary text to chromadb -----

        # ChromaDB에 summary_text를 저장
        chroma_client = get_chroma_client()
        collection = chroma_client.get_or_create_collection(name="user_resume_db")

        metadata = {
            "session_id": state['session_id'],
            "user_id": state['user_id'],
            "type": "resume_summary"
        }

        id_value = f"{state['session_id']}_{state['user_id']}"

        # 기존 데이터 삭제 (덮어씌움)
        collection.delete(ids=[id_value])

        # ChromaDB에 summary_text를 저장 (내용 기반 검색이 필요 없으므로 dummy 임베딩 사용 가능)
        collection.add(
            documents=[state['user_file_summary']],
            metadatas=[metadata],
            ids=[id_value]
        )
        print(f"✅ 이력서 요약 내용을 ChromaDB에 저장 완료 (id={state['session_id']}_{state['user_id']})")

    return state

@traceable(run_type="chain", name="Simple_Chain")
def generate_resume_feedback_node(state: GraphState) -> GraphState:
    """
    이력서  요약된 내용 + 질문 + 기업 뉴스 요약 기반으로
    맞춤형 피드백을 생성하여 state["feedback"] 에 저장
    """
    print("--- 4b. 맞춤형 이력서 피드백 생성 ---")
    
    llm = get_llm()

    # --- 프롬프트 ---
    prompt_template = ChatPromptTemplate.from_template("""
        당신은 전문 이력서 및 포트폴리오 피드백 컨설턴트입니다.

        다음은 사용자의 질문입니다:
        \"\"\"{question}\"\"\"

        
        다음은 이력서의 항목별 요약 내용입니다:
        \"\"\"{resume_summary}\"\"\"

        # 다음은 지원하려는 기업의 최근 뉴스 요약입니다:
        \"\"\"{context}\"\"\"
                                                       
        위 정보를 바탕으로 아래 조건에 맞게 한국어로 구체적으로 피드백을 작성해 주세요:

        질문에 대한 명확한 답변 및 관련 맥락 언급
        뉴스 요약 내용을 반영해 기업 상황과 연계한 인사이트가 있으면 언급
        구체적이고 실질적으로 면접과 준비에 도움이 되는 형태
        적절한 예시 문장을 포함하여 설명

        다른 불필요한 설명은 작성하지 말고, 피드백만 출력하세요.
    """)

    

    # --- 체인 생성 ---
    feedback_chain = prompt_template | llm | StrOutputParser()

    feedback = feedback_chain.invoke({
        "context": "\n---\n".join(state['relevant_chunks']),
        "question": state['question'],
        "resume_summary": state['user_file_summary']
    })

    cleaned_feedback = clean_llm_output(feedback)

    state["answer"] = cleaned_feedback

    print("✅ 맞춤형 이력서 피드백 생성 완료")
    return state


# ==============================================================================
# 4. 그래프 구성 및 실행 함수
# ==============================================================================

def create_workflow():
    """LangGraph 워크플로우를 생성하고 컴파일합니다."""
    graph = StateGraph(GraphState)

    # 노드 등록
    graph.add_node("retrieve_from_chroma", retrieve_from_chroma_node)
    graph.add_node("route_request", route_request_node)
    graph.add_node("get_tavily_snippets", get_tavily_snippets)
    graph.add_node("generate_answer", generate_answer_node)
    # graph.add_node("grade_answer", grade_answer_node)
    graph.add_node("load_and_summarize_resume", load_and_summarize_resume_node)
    graph.add_node("generate_resume_feedback", generate_resume_feedback_node)
    # 그래프 흐름 정의
    graph.set_entry_point("retrieve_from_chroma")

    graph.add_edge("retrieve_from_chroma", "route_request")

    # 라우팅 조건 설정
    graph.add_conditional_edges(
        "route_request",
        lambda state: state["input_type"],
        {
            "qa": "get_tavily_snippets",
            "feedback": "load_and_summarize_resume",
        }
    )

    # Q&A 경로
    graph.add_edge("get_tavily_snippets", "generate_answer")
    graph.add_edge("generate_answer", END)

    # 피드백 경로
    graph.add_edge("load_and_summarize_resume", "generate_resume_feedback")
    graph.add_edge("generate_resume_feedback", END)  # 피드백 생성 후 종료

    # 워크플로우를 컴파일하여 반환
    return graph.compile()
    
# 워크플로우 앱을 한번만 생성
agent_app = create_workflow()
    
    
# ==============================================================================
# 5. 테스트 실행 블록
# ==============================================================================

if __name__ == "__main__":
    # 이 블록은 직접 실행하여 테스트할 때 사용됩니다.
    # FastAPI 서버에서는 이 블록이 실행되지 않습니다.
    

    # --- 시나리오 1: 뉴스 Q&A 테스트 ---
    print("\n" + "="*50)
    print("시나리오 1: 뉴스 Q&A 테스트 시작")
    print("="*50)
    
    # 가정: news_id=101에 해당하는 뉴스가 이미 ChromaDB에 저장되어 있음
    # (사전 작업: 별도의 스크립트로 뉴스를 ChromaDB에 저장해야 함)
    
    qa_input = {
        "question": "SK쉴더스가 제로트러스트 모델로 뭘 하려는 건가요?",
        "news_id": 101, # Spring 서버로부터 받은 뉴스 ID
        "file_path": None,
        "company": "SK쉴더스",  # 회사명 (추후 Tavily 검색에 사용)
        "chat_history": []
    }
    
    try:
        # stream()을 사용하면 각 단계의 출력을 볼 수 있음
        for output in agent_app.stream(qa_input, {"recursion_limit": 10}):
            node_name = list(output.keys())[0]
            node_output = output[node_name]
            print(f"--- 노드 '{node_name}' 실행 완료 ---")
        
        final_state = agent_app.invoke(qa_input)
        

        print("\n[최종 답변]:", final_state.get('answer'))

    except Exception as e:
        print(f"\n[오류 발생]: {e}")


    # --- 시나리오 2: 이력서 피드백 테스트 ---
    print("\n" + "="*50)
    print("시나리오 2: 이력서 피드백 테스트 시작")
    print("="*50)
    
    # 테스트용 이력서 파일 생성
    resume_file = "./file_data/이력서_이준기.pdf"
    
    feedback_input = {
        "question": "제 이력서에서 자기소개서만 피드백 해주세요.",
        "news_id": None,
        "file_path": resume_file,
        "company": "SK쉴더스",
        "chat_history": []
    }

    try:
        final_state = agent_app.invoke(feedback_input)
        print("\n[최종 답변]:", final_state.get('answer'))
    except Exception as e:
        print(f"\n[오류 발생]: {e}")
        
    # 그래프 시각화
    try:
        graph_image_path = "agent_workflow.png"
        with open(graph_image_path, "wb") as f:
            f.write(agent_app.get_graph().draw_mermaid_png())
        print(f"\n✅ LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"그래프 시각화 중 오류 발생: {e}")
        