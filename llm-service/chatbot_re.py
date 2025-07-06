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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.messages import BaseMessage

# Load API key and suppress warnings
load_dotenv()
warnings.filterwarnings("ignore")

# LLM 설정
llm_split = Ollama(model="qwen3:1.7b")
llm_feedback = Ollama(model="qwen3:1.7b")

# --- 상태 정의 ---
class GraphState(TypedDict):
    user_question: Optional[str]
    chat_history: List[BaseMessage]
    file_path: Optional[str]
    pages: Optional[List]
    classified: Optional[List]
    section_map: Optional[Dict]
    vectorstore: Optional[Any]
    company_analysis: Optional[str]
    news_summary: Optional[str]
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

반드시 위 단어 중 하나만 출력하세요.
""")
route_chain = LLMChain(llm=llm_split, prompt=route_prompt)

split_prompt = ChatPromptTemplate.from_template("""
아래는 이력서 또는 포트폴리오의 한 페이지입니다:

"{page_text}"

이 페이지는 어떤 항목에 해당하나요?
(예: 경력, 학력, 프로젝트, 기술, 자기소개 등)

한 단어로 항목을 분류하고, 간단한 이유를 설명해주세요.
""")
chain_split = LLMChain(llm=llm_split, prompt=split_prompt)

# --- 노드 정의 ---
def route_by_input_type(state: GraphState) -> GraphState:
    result = route_chain.run(question=state["user_question"]).strip().lower()
    print(f"🪐 LLM 판단 결과: {result}")

    if "feedback" in result:
        state = load_resume_pdf(state)
        state = classify_by_page(state)
        state = make_section_map(state)
        state = vector_indexing(state)
        state = load_company_analysis(state)
        state = match_and_feedback(state)
    elif "summarize" in result:
        state = summarize_news(state)
    elif "qa" in result:
        state = answer_question(state)
    else:
        raise ValueError(f"지원되지 않는 응답: {result}")

    return state

def load_resume_pdf(state: GraphState) -> GraphState:
    loader = PyPDFLoader(state["file_path"])
    pages = loader.load()
    print(f"✅ {len(pages)} 페이지 로드 완료")
    return {**state, "pages": pages}

def classify_by_page(state: GraphState) -> GraphState:
    results = []
    for idx, page in enumerate(state["pages"]):
        res = chain_split.run(page_text=page.page_content.strip())
        results.append({
            "page": idx + 1,
            "category": res,
            "content": page.page_content.strip()
        })
    return {**state, "classified": results}

def make_section_map(state: GraphState) -> GraphState:
    section_map = defaultdict(list)
    for item in state["classified"]:
        section = item['category'].split(":")[0].strip()
        section_map[section].append(item['content'])
    return {**state, "section_map": section_map}

def vector_indexing(state: GraphState) -> GraphState:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    texts, metadatas = [], []
    for section, contents in state["section_map"].items():
        for content in contents:
            texts.append(content)
            metadatas.append({"section": section})
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas)
    return {**state, "vectorstore": vectorstore}

def load_company_analysis(state: GraphState) -> GraphState:
    return state

def match_and_feedback(state: GraphState) -> GraphState:
    retriever = state["vectorstore"].as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm_feedback, retriever=retriever)
    prompt = f"""
    다음은 한 기업의 뉴스 기사 분석 내용입니다:

    "{state['company_analysis']}"

    이력서/포트폴리오 항목별로 강조할 점, 부족한 점, 보완점을 구체적으로 작성해주세요.
    """
    raw_feedback = qa_chain.run(prompt)
    feedback = clean_llm_output(raw_feedback)
    print("✅ 피드백 생성 완료")
    return {**state, "feedback": feedback}

# def answer_question(state: GraphState) -> GraphState:
#     retriever = state["vectorstore"].as_retriever()
#     qa_chain = RetrievalQA.from_chain_type(llm=llm_feedback, retriever=retriever)
#     result = qa_chain.run(state["user_question"])
#     print("✅ 질문 답변 생성 완료")
#     return {**state, "answer": result}

def answer_question(state: GraphState) -> GraphState:
    if state.get("vectorstore"):
        retriever = state["vectorstore"].as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm_feedback, retriever=retriever)
        raw_result = qa_chain.run(state["user_question"])
    else:
        raw_result = llm_feedback.invoke(state["user_question"])

    result = clean_llm_output(raw_result)
    print("✅ 질문 답변 생성 완료")
    return {**state, "answer": result}

def summarize_news(state: GraphState) -> GraphState:
    summary = f"[뉴스 요약] {state['user_question'][:50]}..."
    print("✅ 뉴스 요약 완료")
    return {**state, "news_summary": summary}

def clean_llm_output(text: str) -> str:
    # <think>...</think> 블록 제거
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 불필요한 연속 줄바꿈 제거
    text = re.sub(r"\n\s*\n", "\n\n", text)
    # 양 끝 공백 제거
    return text.strip()

# --- 실행 함수 ---
def run_langgraph_flow(user_question: str,
                       resume_path: Optional[str] = None,
                       news_full_path: Optional[str] = None,
                       news_summary_path: Optional[str] = None,
                       chat_history: Optional[List[BaseMessage]] = None) -> Dict:
    state: GraphState = {"user_question": user_question, "chat_history": chat_history or []}

    if resume_path and Path(resume_path).exists():
        state["file_path"] = resume_path

    if news_summary_path and Path(news_summary_path).exists():
        with open(news_summary_path, "r", encoding="utf-8") as f:
            news_json = json.load(f)
            state["company_analysis"] = news_json.get("summary") or news_json.get("content")

    if news_full_path and Path(news_full_path).exists():
        with open(news_full_path, "r", encoding="utf-8") as f:
            full_news_text = f.read()
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        vectorstore = FAISS.from_texts([full_news_text], embeddings)
        state["vectorstore"] = vectorstore

    graph = StateGraph(GraphState)

    graph.add_node("router", route_by_input_type)
    graph.add_node("ClassifyPages", classify_by_page)
    graph.add_node("ToSectionMap", make_section_map)
    graph.add_node("VectorIndexing", vector_indexing)
    graph.add_node("LoadCompanyInfo", load_company_analysis)
    graph.add_node("Feedback", match_and_feedback)

    graph.set_entry_point("router")


    # ✅ Conditional edges with CORRECT MAPPING
    # def router_func(state: GraphState) -> str:
    #     next_node = route_by_input_type(state)
    #     print(f"➡️ [router_func] Moving to: {next_node}")
    #     return next_node

    # graph.add_conditional_edges("router", router_func, {
    #     "LoadPDF": "ClassifyPages",
    #     "summarize_news": END,
    #     "answer_question": END
    # })

    graph.add_edge("ClassifyPages", "ToSectionMap")
    graph.add_edge("ToSectionMap", "VectorIndexing")
    graph.add_edge("VectorIndexing", "LoadCompanyInfo")
    graph.add_edge("LoadCompanyInfo", "Feedback")
    graph.add_edge("Feedback", END)

    compiled = graph.compile()
    result = compiled.invoke(state)
    print("✅ 플로우 실행 완료")
    return result