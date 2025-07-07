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
from langchain_ollama import OllamaLLM
from langchain_core.messages import BaseMessage
import re


'''
================================================

실행 방법 (llm-service 디렉토리에서):
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

================================================
'''

# Load API key and suppress warnings
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
warnings.filterwarnings("ignore")

# LLM 설정
# llm_split = OllamaLLM(model="qwen3:1.7b")
llm_split = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.groq.com/openai/v1",  # Groq API 엔드포인트
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7
)
llm_feedback = OllamaLLM(model="qwen3:1.7b")

# --- 상태 정의 ---
class GraphState(TypedDict):
    user_question: Optional[str] # 사용자 질문
    chat_history: List[BaseMessage] # 대화 기록
    file_path: Optional[str] # 첨부파일 경로
    pages: Optional[List] # 첨부파일 페이지 리스트
    classified: Optional[List] # 페이지 분류 결과
    section_map: Optional[Dict] # 섹션별 내용 맵핑
    vectorstore: Optional[Any]  # 벡터 스토어 (FAISS)
    company_analysis: Optional[str] # 기업 분석 내용
    news_summary: Optional[str] # 뉴스 요약 내용
    feedback: Optional[str] # 피드백 내용
    answer: Optional[str] # 질문에 대한 답변 내용

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
    # 필요 없는 선두/후미 공백 제거
    return text.strip()

# --- 프롬프트 설정 ---
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
route_chain = LLMChain(llm=llm_split, prompt=route_prompt)

split_prompt = ChatPromptTemplate.from_template("""
    아래는 이력서의 한 페이지 텍스트입니다.

    =======================
    {page_text}
    =======================

    텍스트를 아래 항목들로 내용을 나누어 각각 해당하는 부분의 원문 내용을 최대한 그대로 추출해 주세요:

    - 인적사항
    - 학력
    - 경력
    - 프로젝트
    - 기술 스택
    - 수상 및 자격증
    - 자기소개

    다음 JSON 배열 형식으로만 출력하세요.

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

    대응되는 항목이 없으면 Unknown 항목으로 추출해 주세요.
    반드시 JSON 배열만 출력하고 다른 설명은 절대 추가하지 마세요.
""")



# LLMChain으로 페이지 분류 체인 생성
chain_split = LLMChain(llm=llm_split, prompt=split_prompt)

# --- 노드 정의 ---
def route_by_input_type(state: GraphState) -> str:
    raw_result = route_chain.run(question=state["user_question"])
    result = clean_llm_output(raw_result).strip().lower()
    print(f"🪐 LLM 판단 결과 : {result} ({type(result)})")

    if "feedback" in result:
        return "feedback"
    elif "qa" in result:
        return "qa"
    else:
        raise ValueError(f"지원되지 않는 응답: {result}")


def router_node(state: GraphState) -> GraphState:
    # Entry point 역할, 상태 그대로 전달만 하면 됨
    return state

def load_resume_pdf(state: GraphState) -> GraphState:
    # PDF 파일을 로드하여 페이지별 객체 리스트 생성
    loader = PyPDFLoader(state["file_path"])
    pages = loader.load()
    print(f"✅ {len(pages)} 페이지 로드 완료")
    return {**state, "pages": pages}

def classify_by_page(state: GraphState) -> GraphState:
    # 각 페이지를 LLM을 사용해 항목별로 분류
    results = []
    for idx, page in enumerate(state["pages"]):
    #     print(f"--- 페이지 {idx + 1} 분류 중 ---")
    #     print(f"페이지 내용:\n{page.page_content.strip()[:1000]}...")  # 처음 1000자만 출력
    #     raw_res = chain_split.invoke({"page_text": page.page_content.strip()})
    #     print(f"🪐 LLM 출력 결과: {raw_res}")

    #     try:
    #         parsed_res = json.loads(raw_res)
    #         for item in parsed_res:
    #             results.append({
    #                 "page": idx + 1,
    #                 "category": item["category"],
    #                 "content": item["content"].strip()
    #             })
    #     except Exception as e:
    #         # 파싱 실패 시 페이지 전체를 unknown으로 저장해 추후 확인 가능
    #         print(f"⚠️ JSON 파싱 실패: {e}")
    #         print(f"⚠️ LLM 원본 출력:\n{raw_res}")
    #         results.append({
    #             "page": idx + 1,
    #             "category": "Unknown",
    #             "content": page.page_content.strip()
    #         })

        paragraphs = [p.strip() for p in page.page_content.strip().split("\n\n+") if p.strip()]

        for p_idx, paragraph in enumerate(paragraphs):
            print(f"--- 페이지 {idx + 1}, 블록 {p_idx + 1} 분류 중 ---")
            print(f"블록 내용: {paragraph[:300]}...")  # 과도한 출력 방지
            raw_res = chain_split.invoke({"page_text": paragraph})
            print(f"🪐 LLM 출력 결과: {raw_res}")

            try:
                parsed_res = json.loads(raw_res["text"] if isinstance(raw_res, dict) and "text" in raw_res else raw_res)
                for item in parsed_res:
                    results.append({
                        "page": idx + 1,
                        "category": item["category"],
                        "content": item["content"].strip()
                    })
            except Exception as e:
                print(f"⚠️ JSON 파싱 실패: {e}")
                print(f"⚠️ LLM 원본 출력:\n{raw_res}")
                results.append({
                    "page": idx + 1,
                    "category": "Unknown",
                    "content": paragraph
                })


    output_path = "./file_data/classified_pages.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 분류 결과가 '{output_path}' 파일로 저장되었습니다.")

    return {**state, "classified": results}


def make_section_map(state: GraphState) -> GraphState:
    # 분류된 페이지 내용을 섹션별로 묶음
    section_map = defaultdict(list)
    for item in state["classified"]:
        section = item["category"].split(":")[0].strip()
        section_map[section].append(item["content"])
    return {**state, "section_map": dict(section_map)}


def vector_indexing(state: GraphState) -> GraphState:
    # HuggingFace 임베딩 + FAISS를 사용해 벡터스토어 생성
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    texts, metadatas = [], []
    for section, contents in state["section_map"].items():
        for content in contents:
            texts.append(content)
            metadatas.append({"section": section})
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas)
    return {**state, "vectorstore": vectorstore}

def load_company_analysis(state: GraphState) -> GraphState:
    # 회사 분석 요약 로드 (이 단계에서 추가 작업은 없지만 구조상 필요)
    return state

def match_and_feedback(state: GraphState) -> GraphState:
    # 뉴스 요약 + 이력서 내용 + 질문 기반으로 LLM이 피드백 생성
    retriever = state["vectorstore"].as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm_feedback, retriever=retriever)

    prompt = f"""
    다음은 사용자 질문입니다:
    \"\"\"{state['user_question']}\"\"\"

    아래는 한 기업의 뉴스 기사 분석 내용입니다:
    \"\"\"{state['company_analysis']}\"\"\"

    그리고 첨부된 이력서 또는 포트폴리오 내용을 바탕으로 한 정보가 포함되어 있습니다.

    뉴스 기사 내용과 첨부 파일의 내용을 모두 고려하여,  
    사용자 질문에 대해 이력서/포트폴리오 항목별로 강조할 점, 부족한 점, 보완할 점을 구체적이고 명확하게 작성해 주세요.
    """
    raw_feedback = qa_chain.run(prompt)
    feedback = clean_llm_output(raw_feedback)
    print("✅ 피드백 생성 완료")
    return {**state, "feedback": feedback}

def answer_question(state: GraphState) -> GraphState:
    # 일반 질문에 대한 답변 생성
    # 벡터스토어가 있으면 Retrieval QA로, 없으면 LLM으로 직접 응답 생성
    if state.get("vectorstore"):
        retriever = state["vectorstore"].as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm_feedback, retriever=retriever)
        raw_result = qa_chain.run(state["user_question"])
    else:
        raw_result = llm_feedback.invoke(state["user_question"])

    result = clean_llm_output(raw_result)
    print("✅ 질문 답변 생성 완료")
    return {**state, "answer": result}


# --- 실행 함수 ---
def run_langgraph_flow(user_question: str,
                       resume_path: Optional[str] = None,
                       news_full_path: Optional[str] = None,
                       news_summary_path: Optional[str] = None,
                       chat_history: Optional[List[BaseMessage]] = None) -> Dict:
    
    # 초기 상태 세팅
    state: GraphState = {"user_question": user_question, "chat_history": chat_history or []}

    # 이력서 경로가 있으면 로드
    if resume_path and Path(resume_path).exists():
        state["file_path"] = resume_path

    # 뉴스 요약 json 경로가 있으면 로드
    if news_summary_path and Path(news_summary_path).exists():
        with open(news_summary_path, "r", encoding="utf-8") as f:
            news_json = json.load(f)
            state["company_analysis"] = news_json.get("summary") or news_json.get("content")

    # 뉴스 전문 경로가 있으면 로드하여 벡터스토어 생성
    if news_full_path and Path(news_full_path).exists():
        with open(news_full_path, "r", encoding="utf-8") as f:
            full_news_text = f.read()
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        vectorstore = FAISS.from_texts([full_news_text], embeddings)
        state["vectorstore"] = vectorstore

    # LangGraph 객체 생성
    graph = StateGraph(GraphState)
    
    # 노드 등록
    # graph.add_node("router", route_by_input_type)
    graph.add_node("router", router_node)
    graph.add_node("LoadPDF", load_resume_pdf)
    graph.add_node("ClassifyPages", classify_by_page)
    graph.add_node("ToSectionMap", make_section_map)
    graph.add_node("VectorIndexing", vector_indexing)
    graph.add_node("LoadCompanyInfo", load_company_analysis)
    graph.add_node("Feedback", match_and_feedback)
    graph.add_node("AnswerQuestion", answer_question)

    graph.set_entry_point("router")

    # 라우팅 조건 설정: qa or feedback
    graph.add_conditional_edges("router", route_by_input_type, {
        "feedback": "LoadPDF",
        "qa": "AnswerQuestion"
    })

    # 피드백 플로우 연결
    graph.add_edge("LoadPDF", "ClassifyPages")
    graph.add_edge("ClassifyPages", "ToSectionMap")
    graph.add_edge("ToSectionMap", "VectorIndexing")
    graph.add_edge("VectorIndexing", "LoadCompanyInfo")
    graph.add_edge("LoadCompanyInfo", "Feedback")
    graph.add_edge("Feedback", END)
    
    # QA 플로우 연결
    graph.add_edge("AnswerQuestion", END)

    # 그래프 시각화
    # app = graph.compile()

    # try:
    #     graph_image_path = "langgraph_structure_nh.png"
    #     with open(graph_image_path, "wb") as f:
    #         f.write(app.get_graph().draw_mermaid_png())
    #     print(f"LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    # except Exception as e:
    #     print(f"그래프 시각화 중 오류 발생: {e}")

    # 그래프 컴파일 및 실행
    compiled = graph.compile()
    result = compiled.invoke(state)
    print("✅ 플로우 실행 완료")
    return result

