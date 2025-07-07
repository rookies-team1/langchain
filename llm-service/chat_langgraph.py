import os
import sys

from langchain_openai import ChatOpenAI

# llm-service 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from typing import TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv
from collections import defaultdict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
import json
from langchain_community.vectorstores import Chroma
from chromadb import chromadb

import re

llm = None
embeddings = None
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# HTTP 클라이언트를 사용하여 Docker 컨테이너로 실행 중인 ChromaDB에 접속
# 'chromadb'는 docker-compose.yml에 정의된 서비스 이름
# chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT) 

# --- 설정 및 초기화 ---
def initialize_models_and_retriever():
    load_dotenv()
    
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url="https://api.groq.com/openai/v1",  # Groq API 엔드포인트
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.7
    )
    
    try:
        embeddings = OllamaEmbeddings(
            model="bge-m3:latest", 
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        # Chroma 벡터 스토어 등록하기
        
        # vectorstore = Chroma(
        #     client=chroma_client,
        #     collection_name="news_collection", # 사용할 컬렉션 이름 지정
        #     embedding_function=embeddings,
        # )
        return llm, embeddings, None
    except Exception as e:
        print(f"임베딩 모델 로드 실패: {e}")
        embeddings = None
    
    return llm, None, None

llm, retriever, news_article = initialize_models_and_retriever()


class GraphState(TypedDict):
    user_question: Optional[str] # 사용자 질문
    company: Optional[str]    # 회사 이름
    chat_history: List[BaseMessage] # 대화 기록
    file_path: Optional[str] # 첨부파일 경로
    pages: Optional[List] # 첨부파일 페이지 리스트
    classified: Optional[List] # 페이지 분류 결과
    section_map: Optional[Dict] # 섹션별 내용 맵핑
    news_vectorstore: Optional[Any] # 뉴스 전문 벡터 스토어 (FAISS)
    resume_vectorstore: Optional[Any] # 첨부파일 벡터 스토어 (FAISS)
    company_analysis: Optional[str] # 기업 분석 내용
    news_summary: Optional[str] # 뉴스 요약 내용
    feedback: Optional[str] # 피드백 내용
    answer: Optional[str] # 질문에 대한 답변 내용
    
    input_type: str # 뉴스 Q&A or 문서 피드백 
    question: str   # 정제된 질문
    uploaded_document: str  # 업로드된 문서 (string or 경로)
    relevant_chunks: List[str]  # 현재 검색기가 바라보고 있는 내용
    is_grounded: bool   # 답변 평가
    news_article: str   # 뉴스 기사 원문
    
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

# --- 라우팅 프롬프트 설정 ---
route_prompt = ChatPromptTemplate.from_template("""
    다음은 사용자의 입력입니다:

    "{question}"

    이 입력은 어떤 작업을 요청하고 있나요?

    - 일반 정보 질문이면 "qa"
    - 첨부 문서를 기반으로 피드백 요청이면 "feedback"

    반드시 위 단어 qa와 feedback 중 하나만 출력하세요.
    절대 다른 단어를 포함하지 마세요.
""")

# LLMChain으로 라우팅 체인 생성
route_chain = route_prompt | llm

split_prompt = ChatPromptTemplate.from_template("""
    아래는 사용자의 이력서 혹은 포트폴리오의 일부 텍스트입니다.

    =======================
    {page_text}
    =======================

    이 텍스트를 아래 항목 중 해당하는 항목에 대응되도록 **원문 내용을 그대로** 추출해 주세요:
    만약 아래 항목 중 어떤 항목에도 딱 맞지 않더라도, **텍스트를 절대 버리지 말고** 적절한 항목 이름을 새로 설정하여 반드시 포함허거나 기타 항목으로 대응시켜 주세요.
    
    - 인적사항
    - 학력
    - 경력
    - 자기소개
    - 프로젝트
    - 대외활동
    - 기술 스택
    - 수상 및 자격증
    - 기타

    아래와 같은 JSON 배열 형식으로만 출력하세요:

    [
        {{
            "category": "학력",
            "content": "한밭대학교 전자전기공학과 ..."
        }},
        {{
            "category": "경력",
            "content": "한화시스템 센터 하계 인턴십 ..."
        }}
    ]

    **규칙:**
    - 반드시 JSON 배열만 출력하고 다른 설명은 절대 출력하지 마세요.
    - 그러나 입력된 텍스트 중 어느 항목에도 해당하지 않는 부분이 있다면 적절한 새로운 항목으로 분류하여 반드시 출력해 주세요. 
""")


# LLMChain으로 페이지 분류 체인 생성
chain_split = split_prompt | llm

# --- 1. 라우팅 노드 ---
def router_node(state: GraphState) -> GraphState:
    # Entry point 역할, 상태 그대로 전달만 하면 됨
    return state

def route_by_input_type(state: GraphState) -> GraphState:
    raw_result = route_chain.invoke({"question": state["user_question"]})
    result = clean_llm_output(raw_result).strip().lower()
    print(f"✅ LLM 분기 판단 결과 : {result}")

    if "feedback" in result:
        state["input_type"] = "feedback"
        return state
    elif "qa" in result:
        state["input_type"] = "qa"
        return state
    else:
        raise ValueError(f"지원되지 않는 응답: {result}")

# --- 2. 뉴스 Q&A ---
def retrieve_chunks_node(state: GraphState):
    print("--- 2a. 관련 뉴스 기사 검색 ---")
    news_article = state['news_article']
    
    # 뉴스 기사 내용을 기반으로 동적으로 FAISS 벡터스토어 생성
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    docs = text_splitter.create_documents([news_article])
    
    # TODO : Chroma 벡터 스토어로 변경
    if embeddings:
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        state['retriever'] = retriever
    else:
        print(" > 임베딩 모델이 초기화되지 않아 검색기를 생성할 수 없습니다.")
        state['relevant_chunks'] = [news_article] # 전체 기사를 컨텍스트로 사용
        return state

    question = state['question']
    documents = retriever.invoke(question)
    state['relevant_chunks'] = [doc.page_content for doc in documents]
    print(f" > ‘{question[:20]}...’에 대해 {len(documents)}개의 관련 문서를 찾았습니다.")
    return state

def generate_answer_node(state: GraphState):
    print("--- 3a. 답변 생성 (대화 맥락 반영) ---")
    prompt = ChatPromptTemplate.from_template(
        """당신은 뉴스 기사 분석가입니다.
        [이전 대화 내용]과 [뉴스 기사 내용]을 바탕으로 [질문]에 대해 한국어로 명확하고 간결하게 답변하세요.
        내용을 찾을 수 없으면 ‘정보를 찾을 수 없습니다’라고 답변하세요.

        [이전 대화 내용]:
        {chat_history}

        [뉴스 기사 내용]:
        {context}

        [질문]:
        {question}
        """
    )
    rag_chain = prompt | llm | StrOutputParser()
    answer = rag_chain.invoke({
        "chat_history": "\n".join([f"{msg.type}: {msg.content}" for msg in state['chat_history']]),
        "context": "\n---\n".join(state['relevant_chunks']),
        "question": state['question']
    })
    state['answer'] = answer
    # print(f" > 생성된 답변: {answer[:30]}...")
    return state

def grade_answer_node(state: GraphState):
    print("--- 4a. 답변 검증 ---")
    prompt = ChatPromptTemplate.from_template(
        """당신은 답변 평가 전문가입니다. 
        주어진 [뉴스 기사 내용]을 볼 때, [생성된 답변]이 [질문]에 대해 사실에 근거하여 올바르게 답변되었는지 평가해주세요. 
        답변이 내용에 근거했다면 ‘yes’, 그렇지 않다면 ‘no’라고만 답해주세요.

        [뉴스 기사 내용]:
        {context}

        [질문]:
        {question}

        [생성된 답변]:
        {answer}
        """
    )
    grading_chain = prompt | llm | StrOutputParser()
    grade = grading_chain.invoke({
        "context": "\n---\n".join(state['relevant_chunks']),
        "question": state['question'],
        "answer": state['answer']
    })
    
    if "yes" in grade.lower():
        state['is_grounded'] = True
        # print(" > 검증 결과: 통과 (내용에 근거함)")
    else:
        state['is_grounded'] = False
        # print(" > 검증 결과: 실패 (내용에 근거하지 않음)")
    return state


# --- 3. 문서 피드백 ---
def load_resume_file(state: GraphState) -> GraphState:
    file_path = state["file_path"]
    pages = []

    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        print(f"✅ PDF {len(pages)} 페이지 로드 완료")
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # 텍스트 파일은 2000자 단위로 페이지로 나누어 처리 (필요 시 조정)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100
        )
        pages = text_splitter.create_documents([text])
        print(f"✅ TXT {len(pages)} 페이지(청크) 로드 완료")
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. pdf 또는 txt만 지원합니다.")

    return {**state, "pages": pages}


def classify_by_page(state: GraphState) -> GraphState:
    results = []
    for idx, page in enumerate(state["pages"]):
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", page.page_content.strip()) if p.strip()]  # \n 2개 이상으로 분리
        
        # 문장 중간 \n 제거
        for i in range(len(paragraphs)):
            paragraphs[i] = re.sub(r'(?<![.!?])\n(?!\n)', ' ', paragraphs[i])
        
        for p_idx, paragraph in enumerate(paragraphs):
            raw_res = chain_split.invoke({"page_text": paragraph})

            try:
                raw_text = raw_res["text"] if isinstance(raw_res, dict) and "text" in raw_res else raw_res
                parsed_res = json.loads(raw_text)
                for item in parsed_res:
                    category = item.get("category", "").strip()
                    content = item.get("content", "").strip()
                    if not category:
                        category = "Uncategorized"  # 적절한 기본 카테고리명으로 변경 가능
                    results.append({
                        "page": idx + 1,
                        "category": category,
                        "content": content
                    })
            except Exception as e:
                print(f"⚠️ JSON 파싱 실패: {e}")
                results.append({
                    "page": idx + 1,
                    "category": "Unknown",
                    "content": paragraph
                })

    # 분류 결과 출력
    output_path = "./file_data/classified_pages.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 분류 결과가 '{output_path}' 파일로 저장되었습니다.")

    return {**state, "classified": results}

def make_section_map(state: GraphState) -> GraphState:
    # 분류된 페이지 내용을 같은 항목별로 묶음
    section_map = defaultdict(list)
    for item in state["classified"]:
        section = item["category"].split(":")[0].strip()
        section_map[section].append(item["content"])
    print("✅ 섹션 맵 생성 완료 section_map:")
    return {**state, "section_map": dict(section_map)}

def vector_indexing(state: GraphState) -> GraphState:
    # HuggingFace 임베딩 + FAISS를 사용해 벡터스토어 생성
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    texts, metadatas = [], []
    for section, contents in state["section_map"].items():
        for content in contents:
            texts.append(content)
            metadatas.append({"section": section})
    resume_vectorstore = FAISS.from_texts(texts, embeddings, metadatas)
    return {**state, "resume_vectorstore": resume_vectorstore}


def load_company_analysis(state: GraphState) -> GraphState:
    # 회사 분석 요약 로드 (이 단계에서 추가 작업은 없지만 구조상 필요)
    return state

def match_and_feedback(state: GraphState) -> GraphState:
    # 뉴스 요약 + 첨부파일 내용 + 질문 기반으로 LLM이 피드백 생성

    retriever_resume = state["resume_vectorstore"].as_retriever()
    feedback_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever_resume)

    # 벡터스토어에서 이력서/포트폴리오 주요 내용 추출
    resume_contents = []
    if state.get("resume_vectorstore"):
        docs = state["resume_vectorstore"].similarity_search(
            state["user_question"],
            k=7
        )
        resume_contents = [doc.page_content for doc in docs]

    resume_text = "\n\n".join(resume_contents)

    # 벡터스토어에서 뉴스 주요 내용 추출
    # news_contents = []
    # if state.get("news_vectorstore"):
    #     docs = state["news_vectorstore"].similarity_search(
    #         state["user_question"],
    #         k=3
    #     )
    #     news_contents = [doc.page_content for doc in docs]

    # news_texts = "\n\n".join(news_contents)

    prompt = f"""
    당신은 사용자의 질문, 관련 뉴스 기사, 첨부된 파일 내용을 모두 통합하여 피드백을 작성하는 전문 분석가입니다.

    다음은 사용자 질문입니다:
    \"\"\"{state['user_question']}\"\"\"

    다음은 해당 기업의 최근 뉴스 기사 요약입니다:
    \"\"\"{state["company_analysis"]}\"\"\"

    다음은 첨부된 이력서 또는 포트폴리오의 주요 내용 요약입니다:
    \"\"\"{resume_text}\"\"\"

    위의 모든 정보를 바탕으로 다음 사항을 충실히 반영해 작성해 주세요:
    1. **질문에 대한 정확한 답변**과 함께 맥락을 구체적으로 설명할 것.
    2. 뉴스 기사 분석 내용을 반영해 사용자 질문과 연관된 인사이트가 있으면 언급할 것.
    3. 첨부 파일 내용(이력서/포트폴리오) 기반으로 강점, 보완할 점, 개선 방향을 항목별로 구체적으로 작성할 것.
    4. 각 항목별로 '강점', '부족한 점', '보완 방안'으로 나누어 깔끔하게 정리할 것.
    5. 구체적이며 명확하고, 실제 면접 또는 준비에 실질적으로 도움이 되는 형태로 작성할 것.
    """

    print("피드백 생성 중...")

    raw_feedback = feedback_chain.invoke({"query": prompt})
    feedback = clean_llm_output(raw_feedback["result"])
    print("✅ 피드백 생성 완료")
    return {**state, "feedback": feedback}


def run_workflow(
                user_question: str,
                resume_path: Optional[str] = None,
                news_content: Optional[str] = None,
                news_summary_path: Optional[str] = None,
                chat_history: Optional[List[BaseMessage]] = None ):
    
    # LangGraph 객체 생성
    graph = StateGraph(GraphState)
    
    # 노드 등록
    # 시작점 (라우터)
    graph.add_node("router", router_node)
    graph.set_entry_point("router")
    
    # Q&A 관련 노드
    graph.add_node("retrieve_chunks", retrieve_chunks_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("grade_answer", grade_answer_node)
    
    # 문서 피드백 관련 노드
    graph.add_node("LoadFile", load_resume_file)
    graph.add_node("ClassifyPages", classify_by_page)
    graph.add_node("ToSectionMap", make_section_map)
    graph.add_node("VectorIndexing", vector_indexing)
    graph.add_node("LoadCompanyInfo", load_company_analysis)
    graph.add_node("Feedback", match_and_feedback)
    
    # 라우팅 조건 설정: qa or feedback
    graph.add_conditional_edges(
        "router",
        lambda state: state["input_type"],
        {
            "feedback": "LoadFile",
            "qa": "retrieve_chunks"
        }
    )
    
    # Q&A 관련 노드들 연결하기
    graph.add_edge("retrieve_chunks", "generate_answer")
    graph.add_edge("generate_answer", "grade_answer")
    graph.add_conditional_edges(
        "grade_answer",
        lambda state: "grounded" if state["is_grounded"] else "not_grounded",
        {
            "grounded": END,
            "not_grounded": END, 
        }
    )
    
    # 문서 피드백 관련 노드들 연결하기
    graph.add_edge("LoadFile", "ClassifyPages")
    graph.add_edge("ClassifyPages", "ToSectionMap")
    graph.add_edge("ToSectionMap", "VectorIndexing")
    graph.add_edge("VectorIndexing", "LoadCompanyInfo")
    graph.add_edge("LoadCompanyInfo", "Feedback")
    graph.add_edge("Feedback", END)
    
    app = graph.compile()
    return app
    
    
if __name__ == "__main__":
    app = run_workflow(
        user_question="",
        resume_path="./file_data/이력서_이준기.pdf",
        news_content=None,
        news_summary_path="./news_data/news_sample1_summary.json",
        chat_history=None
    )
    
     # 그래프 시각화
    try:
        graph_image_path = "langgraph_structure_new.png"
        with open(graph_image_path, "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print(f"LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"그래프 시각화 중 오류 발생: {e}")
