import os
import json
import warnings
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# Load API key and suppress warnings
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
warnings.filterwarnings("ignore")

# LLM 설정
llm_config = {
    "api_key": OPENAI_API_KEY,
    "base_url": "https://api.groq.com/openai/v1",
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "temperature": 0.7
}
llm_split = ChatOpenAI(**llm_config)
llm_feedback = ChatOpenAI(**llm_config)

# Prompt 설정
prompt_template = ChatPromptTemplate.from_template("""
아래는 이력서 또는 포트폴리오의 한 페이지입니다:

"{page_text}"

이 페이지는 어떤 항목에 해당하나요?
(예: 경력, 학력, 프로젝트, 기술, 자기소개, 수상/자격증, 연락처, 기타)

그 항목을 한 단어로 분류하고, 간단히 이유를 설명해주세요.
""")
chain_split = LLMChain(llm=llm_split, prompt=prompt_template)

# 노드 정의
def load_resume_pdf(state):
    loader = PyPDFLoader(state["file_path"])
    return {**state, "pages": loader.load()}

def classify_by_page(state):
    results = []
    for idx, page in enumerate(state["pages"]):
        res = chain_split.run(page_text=page.page_content.strip())
        results.append({
            "page": idx + 1,
            "category": res,
            "content": page.page_content.strip()
        })
    return {**state, "classified": results}

def make_section_map(state):
    section_map = defaultdict(list)
    for item in state["classified"]:
        section = item['category'].split(":")[0].strip()
        section_map[section].append(item['content'])
    return {**state, "section_map": section_map}

def vector_indexing(state):
    file_path = state["file_path"]
    filename = os.path.splitext(os.path.basename(file_path))[0]
    save_dir = f"./vectorstore/{filename}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if os.path.exists(os.path.join(save_dir, "index.faiss")):
        print(f"[벡터 인덱스 로드됨]: {save_dir}")
        embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        vectorstore = FAISS.load_local(save_dir, embeddings)
        return {**state, "vectorstore": vectorstore}

    texts, metadatas = [], []
    for section, contents in state["section_map"].items():
        for content in contents:
            texts.append(content)
            metadatas.append({"section": section})

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas)
    vectorstore.save_local(save_dir)
    print(f"[벡터 인덱스 저장 완료]: {save_dir}")
    return {**state, "vectorstore": vectorstore}

def load_company_analysis(state):
    return state

def match_and_feedback(state):
    retriever = state["vectorstore"].as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm_feedback, retriever=retriever)
    prompt = f"""
    다음은 한 기업의 뉴스 기사 분석 내용입니다:

    "{state['company_analysis']}"

    이력서/포트폴리오 항목을 기준으로, 강조할 점 / 부족한 점 / 보완점을 항목별로 정리해주세요.
    뉴스 기사 분석 내용을 반영하여, 이력서/포트폴리오의 각 항목에 대해 강조할 점 / 부족한 점 / 보완점을 정리해주세요:
    """
    return {**state, "feedback": qa_chain.run(prompt)}

def process_resume_with_news(resume_path: str, news_path: str) -> str:
    # 파일 존재 확인
    if not Path(resume_path).exists():
        raise FileNotFoundError(f"이력서 파일을 찾을 수 없습니다: {resume_path}")
    if not Path(news_path).exists():
        raise FileNotFoundError(f"뉴스 요약 파일을 찾을 수 없습니다: {news_path}")

    # 뉴스 요약 로드
    with open(news_path, "r", encoding="utf-8") as f:
        news_json = json.load(f)
        company_analysis = news_json.get("summary") or news_json.get("content") or ""

    # LangGraph 구성
    graph = StateGraph(state_schema=dict)
    graph.add_node("LoadPDF", load_resume_pdf)
    graph.add_node("ClassifyPages", classify_by_page)
    graph.add_node("ToSectionMap", make_section_map)
    graph.add_node("VectorIndexing", vector_indexing)
    graph.add_node("LoadCompanyInfo", load_company_analysis)
    graph.add_node("Feedback", match_and_feedback)

    graph.set_entry_point("LoadPDF")
    graph.add_edge("LoadPDF", "ClassifyPages")
    graph.add_edge("ClassifyPages", "ToSectionMap")
    graph.add_edge("ToSectionMap", "VectorIndexing")
    graph.add_edge("VectorIndexing", "LoadCompanyInfo")
    graph.add_edge("LoadCompanyInfo", "Feedback")
    graph.add_edge("Feedback", END)

    compiled_graph = graph.compile()

    # 실행
    final_state = compiled_graph.invoke({
        "file_path": resume_path,
        "company_analysis": company_analysis
    })
    return final_state["feedback"]
