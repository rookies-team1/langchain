# poetry run python ..

import os

from typing import TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv
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
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from chromadb import chromadb
import re

# ==============================================================================
# 1. 초기화 및 설정
# ==============================================================================

llm = None
embeddings = None
chroma_client = None

collection_name = "news_vector_db"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_llm():
    """LLM 인스턴스를 가져옵니다. 없으면 생성합니다."""
    global llm
    if llm is None:
        load_dotenv()
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        # llm = ChatOpenAI(
        #     api_key=OPENAI_API_KEY,
        #     base_url="https://api.groq.com/openai/v1",  # Groq API 엔드포인트
        #     model="meta-llama/llama-4-scout-17b-16e-instruct",
        #     temperature=0.7
        # )
    return llm

def get_embeddings():
    """임베딩 모델 인스턴스를 가져옵니다. 없으면 생성합니다."""
    global embeddings
    if embeddings is None:
        load_dotenv()
        try:
            # ChromaDB와 같은 영구적인 저장소를 사용할 것이므로, 일관된 임베딩 모델 사용이 중요
            embeddings = OllamaEmbeddings(
                model="bge-m3:567m", 
                base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
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
        CHROMA_HOST = os.getenv("VECTOR_DB_HOST", "chroma")
        CHROMA_PORT = int(os.getenv("VECTOR_DB_PORT", "8001")) # ChromaDB 기본 포트는 8000이나, docker-compose 예시에서 8001로 설정
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return chroma_client

# ==============================================================================
# 2. Graph State 정의
# ==============================================================================

class GraphState(TypedDict):
    # 입력 값
    user_question: str
    news_id: int  # 뉴스 식별자
    file_path: Optional[str]
    chat_history: List[BaseMessage]

    # 그래프 내부에서 관리되는 값
    input_type: str  # 'qa' or 'feedback'
    question: str  # 재구성된 질문
    
    # QA 경로 관련 상태
    retriever: Optional[Any] # Retriever 객체 저장 (수정: 상태에 retriever 추가)
    relevant_chunks: List[str]
    answer: str
    is_grounded: bool

    # Feedback 경로 관련 상태
    pages: Optional[List]
    user_file_summary: Optional[str]
    feedback: str

# ==============================================================================
# 3. LangGraph 노드 함수 정의
# ==============================================================================

def clean_llm_output(text: str) -> str:
    # LLM의 출력물에서 필요 없는 태그/마크다운/불필요한 줄바꿈 정리

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


# --- 라우팅 노드 ---
def route_request_node(state: GraphState) -> dict:
    """사용자 질문을 분석하여 다음 단계를 결정하는 라우터"""
    print("--- 1. 요청 라우팅 ---")
    llm = get_llm()
    route_prompt = ChatPromptTemplate.from_template(
        """사용자 질문 '{question}'은 다음 중 어떤 유형에 가장 가깝습니까?
        - 뉴스 기사에 대한 질문: 'qa'
        - 첨부된 문서(이력서/포트폴리오)에 대한 피드백 요청: 'feedback'
        답변은 반드시 'qa' 또는 'feedback' 단어 하나만 포함해야 합니다."""
    )
    routing_chain = route_prompt | llm | StrOutputParser()
    result = routing_chain.invoke({"question": state["user_question"]})
    
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
def retrieve_from_chroma_node(state: GraphState):
    """(성능 개선) ChromaDB에서 news_id를 필터링하여 관련 청크를 검색"""
    print(f"--- 2a. ChromaDB에서 뉴스 검색 (news_id: {state['news_id']}) ---")
    embeddings = get_embeddings()
    chroma_client = get_chroma_client()
    
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name, # 사전에 뉴스가 저장된 컬렉션
        embedding_function=embeddings,
    )
    
    # news_id를 메타데이터 필터로 사용하여 해당 뉴스 기사 내에서만 검색
    retriever = vectorstore.as_retriever(
        search_kwargs={'filter': {'news_id': str(state['news_id'])}, "k": 3}
    )
    
    # 재구성된 질문 또는 원본 질문을 사용 (이 예제에서는 원본 사용)
    question = state['user_question']
    # documents = retriever.invoke(question)
    try:
        documents = retriever.get_relevant_documents(question)
    except AttributeError:
        # fallback for retriever implementations that use 'invoke'
        documents = retriever.invoke(question)
    
    if not documents:
        print(f"⚠️ news_id '{state['news_id']}'에 해당하는 문서를 ChromaDB에서 찾을 수 없습니다.")
        # fallback: state에 news_content가 있다면 그것을 사용 (API 설계에 따라)
        # 이 예제에서는 빈 리스트로 처리
        state['relevant_chunks'] = []
    else:
        state['relevant_chunks'] = [doc.page_content for doc in documents]
        print(f"✅ ‘{question[:20]}...’에 대해 {len(documents)}개의 관련 문서를 찾았습니다.")
    return state
  
  
def generate_answer_node(state: GraphState):
    print("--- 3a. 답변 생성 ---")
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """[뉴스 기사 내용]을 바탕으로 [질문]에 대해 한국어로 명확하고 간결하게 답변하세요.
        [뉴스 기사 내용]: {context}
        [질문]: {question}"""
    )
    rag_chain = prompt | llm | StrOutputParser()
    
    if not state['relevant_chunks']:
        answer = "관련 정보를 찾을 수 없습니다. 뉴스 ID를 확인해주세요."
    else:
        answer = rag_chain.invoke({
            "context": "\n---\n".join(state['relevant_chunks']),
            "question": state['user_question']
        })
    state['answer'] = clean_llm_output(answer)
    return state

def grade_answer_node(state: GraphState):
    print("--- 4a. 답변 검증 ---")
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
        "question": state['user_question'],
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
def load_and_summarize_resume_node(state: GraphState):
    print("--- 2b. 이력서 로드 및 요약 ---")
    file_path = state["file_path"]
    if not file_path or not os.path.exists(file_path):
        raise ValueError("피드백을 위한 파일 경로가 유효하지 않습니다.")

    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
    else: # txt
        loader = TextLoader(file_path, encoding='utf-8')
        pages = loader.load_and_split()
    
    full_text = " ".join([page.page_content for page in pages])

    # (비용/시간 개선) 전체 텍스트를 한번에 요약하도록 변경
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        "다음 이력서/포트폴리오의 핵심 역량과 프로젝트 경험을 3~5 문장으로 요약해줘.\n\n{text}")
    summarization_chain = prompt | llm | StrOutputParser()
    summary = summarization_chain.invoke({"text": full_text})
    
    state['user_file_summary'] = f"{clean_llm_output(summary)}"
    print("✅ 이력서 요약 완료")
    return state

def generate_resume_feedback_node(state: GraphState) -> GraphState:
    """
    이력서  요약된 내용 + 질문 + 기업 뉴스 요약 기반으로
    맞춤형 피드백을 생성하여 state["feedback"] 에 저장
    """
    print("--- 3b. 맞춤형 이력서 피드백 생성 ---")
    
    llm = get_llm()

    # --- 프롬프트 ---
    prompt_template = ChatPromptTemplate.from_template("""
        당신은 전문 이력서 및 포트폴리오 피드백 컨설턴트입니다.

        다음은 사용자의 질문입니다:
        \"\"\"{question}\"\"\"

        
        다음은 이력서의 항목별 요약 내용입니다:
        \"\"\"{resume_summary}\"\"\"

        위 정보를 바탕으로 아래 조건에 맞게 한국어로 구체적으로 피드백을 작성해 주세요:

        질문에 대한 명확한 답변 및 관련 맥락 언급
        뉴스 요약 내용을 반영해 기업 상황과 연계한 인사이트가 있으면 언급
        구체적이고 실질적으로 면접과 준비에 도움이 되는 형태

        다른 불필요한 설명은 작성하지 말고, 피드백만 출력하세요.
    """)

    # 다음은 지원하려는 기업의 최근 뉴스 요약입니다:
    # \"\"\"{context}\"\"\"

    # --- 체인 생성 ---
    feedback_chain = prompt_template | llm | StrOutputParser()

    feedback = feedback_chain.invoke({
        # "context": "\n---\n".join(state['relevant_chunks']),
        "question": state['user_question'],
        "resume_summary": state['user_file_summary']
    })

    cleaned_feedback = clean_llm_output(feedback)

    state["feedback"] = cleaned_feedback

    print("✅ 맞춤형 이력서 피드백 생성 완료")
    return state


# ==============================================================================
# 4. 그래프 구성 및 실행 함수
# ==============================================================================

def create_workflow():
    """LangGraph 워크플로우를 생성하고 컴파일합니다."""
    graph = StateGraph(GraphState)

    # 노드 등록
    graph.add_node("route_request", route_request_node)
    graph.add_node("retrieve_from_chroma", retrieve_from_chroma_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("grade_answer", grade_answer_node)
    graph.add_node("load_and_summarize_resume", load_and_summarize_resume_node)
    graph.add_node("generate_resume_feedback", generate_resume_feedback_node)
    # 그래프 흐름 정의
    graph.set_entry_point("route_request")

    # 라우팅 조건 설정
    graph.add_conditional_edges(
        "route_request",
        lambda state: state["input_type"],
        {
            "qa": "retrieve_from_chroma",
            "feedback": "load_and_summarize_resume",
        }
    )

    # Q&A 경로
    graph.add_edge("retrieve_from_chroma", "generate_answer")
    graph.add_edge("generate_answer", "grade_answer")
    graph.add_conditional_edges(
        "grade_answer",
        lambda state: "grounded" if state.get("is_grounded", True) else "not_grounded",
        {
            "grounded": END,
            "not_grounded": "generate_answer"  # 실패 시 답변 재생성 (Self-Correction)
        }
    )

    # 피드백 경로
    graph.add_edge("load_and_summarize_resume", "generate_resume_feedback")
    # graph.add_conditional_edges(
    #     "generate_resume_feedback",
    #     lambda state: "feedback_generated" if state.get("feedback") else "feedback_failed", # 피드백이 생성되었는지 확인
    #     {
    #         "feedback_generated": END,
    #         "feedback_failed": "load_and_summarize_resume"  # 실패 시 종료 (추후 개선 가능)
    #     }
    # )
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
        "user_question": "SK쉴더스가 제로트러스트 모델로 뭘 하려는 건가요?",
        "news_id": 101, # Spring 서버로부터 받은 뉴스 ID
        "file_path": None,
        "chat_history": []
    }
    
    try:
        # stream()을 사용하면 각 단계의 출력을 볼 수 있음
        for output in agent_app.stream(qa_input, {"recursion_limit": 5}):
            node_name = list(output.keys())[0]
            node_output = output[node_name]
            print(f"--- 노드 '{node_name}' 실행 완료 ---")
            # print(f"상태: {node_output}\n")
        
        final_state = agent_app.invoke(qa_input)
        print("\n[최종 답변]:", final_state.get('answer'))

    except Exception as e:
        print(f"\n[오류 발생]: {e}")


    # --- 시나리오 2: 이력서 피드백 테스트 ---
    print("\n" + "="*50)
    print("시나리오 2: 이력서 피드백 테스트 시작")
    print("="*50)
    
    # 테스트용 이력서 파일 생성
    resume_file = "test_resume.txt"
    with open(resume_file, "w", encoding="utf-8") as f:
        f.write("이준기\nPython, Java 개발 경험. LangChain 프로젝트 수행.")
    
    feedback_input = {
        "user_question": "제 이력서에서 자기소개서만 피드백 해주세요.",
        "news_id": None,
        "file_path": resume_file,
        "chat_history": []
    }

    try:
        final_state = agent_app.invoke(feedback_input)
        print("\n[최종 답변]:", final_state.get('feedback'))
    except Exception as e:
        print(f"\n[오류 발생]: {e}")
        
    # 그래프 시각화
    # try:
    #     graph_image_path = "agent_workflow.png"
    #     with open(graph_image_path, "wb") as f:
    #         f.write(agent_app.get_graph().draw_mermaid_png())
    #     print(f"\n✅ LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    # except Exception as e:
    #     print(f"그래프 시각화 중 오류 발생: {e}")
        
