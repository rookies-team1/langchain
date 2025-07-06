import os
import json
import warnings
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict
from typing import TypedDict, Optional, List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama



# Load API key and suppress warnings
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
warnings.filterwarnings("ignore")

# LLM 설정
# llm_config = {
#     "api_key": OPENAI_API_KEY,
#     "base_url": "https://api.groq.com/openai/v1",
#     "model": "meta-llama/llama-4-scout-17b-16e-instruct",
#     "temperature": 0.7
# }
# llm_split = ChatGroq(**llm_config)
# llm_feedback = ChatGroq(**llm_config)

llm_split = Ollama(model="qwen3:1.7b")
llm_feedback = Ollama(model="qwen3:1.7b")

# --- 상태 정의 ---
class GraphState(TypedDict):
    company_name: Optional[str]
    news_articles: Optional[List[Dict]]
    news_summary: Optional[str]
    chat_history: List[BaseMessage]
    uploaded_file_content: Optional[str]
    enhanced_resume: Optional[str]
    user_question: Optional[str]
    next_node: Optional[str]
    file_path: Optional[str]
    section_map: Optional[Dict]
    classified: Optional[List]
    pages: Optional[List]
    vectorstore: Optional[Any]
    company_analysis: Optional[str]
    feedback: Optional[str]
    answer: Optional[str]

# --- 프롬프트 설정 ---
route_prompt = ChatPromptTemplate.from_template("""
다음은 사용자의 입력입니다:

"{question}"

이 입력은 어떤 작업을 요청하고 있나요?

- 일반 정보 질문이면 "qa"
- 첨부 문서를 기반으로 피드백 요청이면 "feedback"
- 뉴스 요약 요청이면 "summarize"

이 중 하나로만 답해주세요.
""")
route_chain = LLMChain(llm=llm_split, prompt=route_prompt)

# # --- 분기 노드 ---
# def route_by_input_type(state: GraphState) -> str:
#     print("🔍 LLM으로 next_node 판단 중...")
#     result = route_chain.run(question=state["user_question"]).strip().lower()

#     if result == "qa":
#         return "answer_question"
#     elif result == "feedback":
#         return "LoadPDF"
#     elif result == "summarize":
#         return "summarize_news"
#     else:
#         raise ValueError(f"지원되지 않는 next_node 응답: {result}")

# # --- 노드 정의 ---
# def load_resume_pdf(state: GraphState) -> GraphState:
#     loader = PyPDFLoader(state["file_path"])
#     return {**state, "pages": loader.load()}

# def classify_by_page(state: GraphState) -> GraphState:
#     results = []
#     for idx, page in enumerate(state["pages"]):
#         res = chain_split.run(page_text=page.page_content.strip())
#         results.append({
#             "page": idx + 1,
#             "category": res,
#             "content": page.page_content.strip()
#         })
#     return {**state, "classified": results}

# def make_section_map(state: GraphState) -> GraphState:
#     section_map = defaultdict(list)
#     for item in state["classified"]:
#         section = item['category'].split(":")[0].strip()
#         section_map[section].append(item['content'])
#     return {**state, "section_map": section_map}

# def vector_indexing(state: GraphState) -> GraphState:
#     file_path = state["file_path"]
#     filename = os.path.splitext(os.path.basename(file_path))[0]
#     save_dir = f"./vectorstore/{filename}"
#     Path(save_dir).mkdir(parents=True, exist_ok=True)

#     embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
#     if os.path.exists(os.path.join(save_dir, "index.faiss")):
#         print(f"[로드됨] {save_dir}")
#         vectorstore = FAISS.load_local(save_dir, embeddings)
#     else:
#         texts, metadatas = [], []
#         for section, contents in state["section_map"].items():
#             for content in contents:
#                 texts.append(content)
#                 metadatas.append({"section": section})
#         vectorstore = FAISS.from_texts(texts, embeddings, metadatas)
#         vectorstore.save_local(save_dir)
#         print(f"[저장 완료] {save_dir}")

#     return {**state, "vectorstore": vectorstore}

# def load_company_analysis(state: GraphState) -> GraphState:
#     return state

# def match_and_feedback(state: GraphState) -> GraphState:
#     retriever = state["vectorstore"].as_retriever()
#     qa_chain = RetrievalQA.from_chain_type(llm=llm_feedback, retriever=retriever)
#     prompt = f"""
#     다음은 한 기업의 뉴스 기사 분석 내용입니다:

#     "{state['company_analysis']}"

#     이력서/포트폴리오 항목별로, 강조할 점 / 부족한 점 / 보완점을 정리해주세요.
#     """
#     return {**state, "feedback": qa_chain.run(prompt)}

# def answer_question(state: GraphState) -> GraphState:
#     retriever = state["vectorstore"].as_retriever()
#     qa_chain = RetrievalQA.from_chain_type(llm=llm_feedback, retriever=retriever)
#     result = qa_chain.run(state["user_question"])
#     return {**state, "answer": result}

# def summarize_news(state: GraphState) -> GraphState:
#     return {**state, "news_summary": f"[요약 결과] 뉴스 요약 요청됨: {state['user_question']}"}

# --- 이력서 페이지 분류용 LLM 설정 ---
split_prompt = ChatPromptTemplate.from_template("""
아래는 이력서 또는 포트폴리오의 한 페이지입니다:

"{page_text}"

이 페이지는 어떤 항목에 해당하나요?
(예: 경력, 학력, 프로젝트, 기술, 자기소개 등)

한 단어로 항목을 분류하고, 간단한 이유를 설명해주세요.
""")
chain_split = LLMChain(llm=llm_split, prompt=split_prompt)

# # --- 실행 함수 ---
# def run_langgraph_flow(
#     user_question: str,
#     resume_path: Optional[str] = None,
#     news_full_path: Optional[str] = None,
#     news_summary_path: Optional[str] = None,
#     chat_history: Optional[List[BaseMessage]] = None,
# ) -> Dict:
#     state: GraphState = {
#         "user_question": user_question,
#         "chat_history": chat_history or []
#     }

#     if resume_path and Path(resume_path).exists():
#         state["file_path"] = resume_path
#         state["uploaded_file_content"] = "uploaded"

#     if news_summary_path and Path(news_summary_path).exists():
#         with open(news_summary_path, "r", encoding="utf-8") as f:
#             news_json = json.load(f)
#             state["company_analysis"] = news_json.get("summary") or news_json.get("content")

#     if news_full_path and Path(news_full_path).exists():
#         with open(news_full_path, "r", encoding="utf-8") as f:
#             full_news_text = f.read()
#         embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
#         vectorstore = FAISS.from_texts([full_news_text], embeddings)
#         state["vectorstore"] = vectorstore

#     graph = StateGraph(GraphState)

#     # 노드 등록
#     graph.add_node("router", route_by_input_type)
#     graph.add_node("answer_question", answer_question)
#     graph.add_node("summarize_news", summarize_news)

#     # 이력서 분석 흐름
#     graph.add_node("LoadPDF", load_resume_pdf)
#     graph.add_node("ClassifyPages", classify_by_page)
#     graph.add_node("ToSectionMap", make_section_map)
#     graph.add_node("VectorIndexing", vector_indexing)
#     graph.add_node("LoadCompanyInfo", load_company_analysis)
#     graph.add_node("Feedback", match_and_feedback)

#     # 라우터 설정
#     graph.set_entry_point("router")
#     graph.add_conditional_edges("router", route_by_input_type, {
#         "answer_question": END,
#         "summarize_news": END,
#         "LoadPDF": "ClassifyPages"
#     })

#     graph.add_edge("ClassifyPages", "ToSectionMap")
#     graph.add_edge("ToSectionMap", "VectorIndexing")
#     graph.add_edge("VectorIndexing", "LoadCompanyInfo")
#     graph.add_edge("LoadCompanyInfo", "Feedback")
#     graph.add_edge("Feedback", END)

#     # 실행
#     compiled = graph.compile()
#     result = compiled.invoke(state)
#     return result

# --- 분기 노드 ---
def route_by_input_type(state: GraphState) -> str:
    print("🔹 [route_by_input_type] 함수 진입")
    print("🔍 LLM으로 next_node 판단 중...")
    result = route_chain.run(question=state["user_question"]).strip().lower()
    print(f"🪐 next_node 결정: {result}")

    if result == "qa":
        return "answer_question"
    elif result == "feedback":
        return "LoadPDF"
    elif result == "summarize":
        return "summarize_news"
    else:
        raise ValueError(f"지원되지 않는 next_node 응답: {result}")

# --- 노드 정의 ---
def load_resume_pdf(state: GraphState) -> GraphState:
    print("🔹 [load_resume_pdf] 이력서 PDF 로드 중...")
    loader = PyPDFLoader(state["file_path"])
    pages = loader.load()
    print(f"✅ {len(pages)} 페이지 로드 완료")
    return {**state, "pages": pages}

def classify_by_page(state: GraphState) -> GraphState:
    print("🔹 [classify_by_page] 페이지별 분류 중...")
    results = []
    for idx, page in enumerate(state["pages"]):
        print(f"🗂️ 페이지 {idx + 1} 분류 중...")
        res = chain_split.run(page_text=page.page_content.strip())
        results.append({
            "page": idx + 1,
            "category": res,
            "content": page.page_content.strip()
        })
        print(f"✅ 페이지 {idx + 1} 분류 결과: {res}")
    return {**state, "classified": results}

def make_section_map(state: GraphState) -> GraphState:
    print("🔹 [make_section_map] 섹션별로 내용 매핑 중...")
    section_map = defaultdict(list)
    for item in state["classified"]:
        section = item['category'].split(":")[0].strip()
        section_map[section].append(item['content'])
    print(f"✅ 섹션 맵핑 완료: {list(section_map.keys())}")
    return {**state, "section_map": section_map}

def vector_indexing(state: GraphState) -> GraphState:
    print("🔹 [vector_indexing] 벡터스토어 인덱싱 중...")
    file_path = state["file_path"]
    filename = os.path.splitext(os.path.basename(file_path))[0]
    save_dir = f"./vectorstore/{filename}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    if os.path.exists(os.path.join(save_dir, "index.faiss")):
        print(f"📂 기존 인덱스 로드: {save_dir}")
        vectorstore = FAISS.load_local(save_dir, embeddings)
    else:
        texts, metadatas = [], []
        for section, contents in state["section_map"].items():
            for content in contents:
                texts.append(content)
                metadatas.append({"section": section})
        print(f"💾 새로운 인덱스 생성 및 저장: {save_dir}")
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas)
        vectorstore.save_local(save_dir)
    print("✅ 인덱싱 완료")
    return {**state, "vectorstore": vectorstore}

def load_company_analysis(state: GraphState) -> GraphState:
    print("🔹 [load_company_analysis] 함수 진입")
    return state

def match_and_feedback(state: GraphState) -> GraphState:
    print("🔹 [match_and_feedback] 피드백 생성 중...")
    retriever = state["vectorstore"].as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm_feedback, retriever=retriever)
    prompt = f"""
    다음은 한 기업의 뉴스 기사 분석 내용입니다:

    "{state['company_analysis']}"

    이력서/포트폴리오 항목별로, 강조할 점 / 부족한 점 / 보완점을 정리해주세요.
    """
    feedback = qa_chain.run(prompt)
    print("✅ 피드백 생성 완료")
    return {**state, "feedback": feedback}

def answer_question(state: GraphState) -> GraphState:
    print("🔹 [answer_question] 질문 답변 중...")
    retriever = state["vectorstore"].as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm_feedback, retriever=retriever)
    result = qa_chain.run(state["user_question"])
    print("✅ 답변 생성 완료")
    return {**state, "answer": result}

def summarize_news(state: GraphState) -> GraphState:
    print("🔹 [summarize_news] 뉴스 요약 중...")
    summary = f"[요약 결과] 뉴스 요약 요청됨: {state['user_question']}"
    print("✅ 요약 완료")
    return {**state, "news_summary": summary}

# --- 실행 함수 ---
def run_langgraph_flow(
    user_question: str,
    resume_path: Optional[str] = None,
    news_full_path: Optional[str] = None,
    news_summary_path: Optional[str] = None,
    chat_history: Optional[List[BaseMessage]] = None,
) -> Dict:
    print("🚀 [run_langgraph_flow] 실행 시작")
    state: GraphState = {
        "user_question": user_question,
        "chat_history": chat_history or []
    }

    if resume_path and Path(resume_path).exists():
        print(f"📄 이력서 파일 감지: {resume_path}")
        state["file_path"] = resume_path
        state["uploaded_file_content"] = "uploaded"

    if news_summary_path and Path(news_summary_path).exists():
        print(f"📰 뉴스 요약 파일 감지: {news_summary_path}")
        with open(news_summary_path, "r", encoding="utf-8") as f:
            news_json = json.load(f)
            state["company_analysis"] = news_json.get("summary") or news_json.get("content")

    if news_full_path and Path(news_full_path).exists():
        print(f"📰 뉴스 전문 파일 감지 및 인덱싱: {news_full_path}")
        with open(news_full_path, "r", encoding="utf-8") as f:
            full_news_text = f.read()
        # embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        vectorstore = FAISS.from_texts([full_news_text], embeddings)
        state["vectorstore"] = vectorstore

    print("🛠️ 그래프 빌드 중...")
    graph = StateGraph(GraphState)

    # 노드 등록
    graph.add_node("router", route_by_input_type)
    graph.add_node("answer_question", answer_question)
    graph.add_node("summarize_news", summarize_news)

    # 이력서 분석 흐름
    graph.add_node("LoadPDF", load_resume_pdf)
    graph.add_node("ClassifyPages", classify_by_page)
    graph.add_node("ToSectionMap", make_section_map)
    graph.add_node("VectorIndexing", vector_indexing)
    graph.add_node("LoadCompanyInfo", load_company_analysis)
    graph.add_node("Feedback", match_and_feedback)

    # 라우터 설정
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_by_input_type, {
        "answer_question": END,
        "summarize_news": END,
        "LoadPDF": "ClassifyPages"
    })

    graph.add_edge("ClassifyPages", "ToSectionMap")
    graph.add_edge("ToSectionMap", "VectorIndexing")
    graph.add_edge("VectorIndexing", "LoadCompanyInfo")
    graph.add_edge("LoadCompanyInfo", "Feedback")
    graph.add_edge("Feedback", END)

    # 실행
    print("🏁 그래프 실행 시작")
    compiled = graph.compile()
    result = compiled.invoke(state)
    print("✅ 그래프 실행 완료")
    return result
