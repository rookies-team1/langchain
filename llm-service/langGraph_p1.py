'''
================================================

실행 방법
poetry run python llm-service/langGraph_p1.py

================================================
'''

import os
import sys

# llm-service 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from typing import List, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from modules.load_llm import load_llm

# --- 1. 설정 및 초기화 ---
def initialize_models_and_retriever():
    load_dotenv()
    
    llm = load_llm()
    
    try:
        embeddings = OllamaEmbeddings(
            model="bge-m3:latest", 
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
    except Exception as e:
        print(f"임베딩 모델 로드 실패: {e}")
        embeddings = None

    try:
        script_dir = os.path.dirname(__file__)
        data_path = os.path.join(script_dir, "test_data", "data.txt")
        loader = TextLoader(data_path, encoding="utf-8")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        
        if embeddings:
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            return llm, retriever, "\n".join([doc.page_content for doc in documents])
    except FileNotFoundError:
        print(f"오류: {data_path} 파일을 찾을 수 없습니다.")
    
    return llm, None, None

llm, retriever, news_article = initialize_models_and_retriever()

# --- 2. LangGraph 상태 정의 ---
class GraphState(TypedDict):
    user_input: str
    input_type: str
    question: str
    uploaded_document: str
    relevant_chunks: List[str]
    answer: str
    feedback: str
    is_grounded: bool
    news_article: str
    chat_history: List[BaseMessage]

# --- 3. LangGraph 노드 정의 ---
def classify_input_node(state: GraphState):
    print("--- 1. 입력 유형 분류 ---")
    user_input = state['user_input']
    if "지원서" in user_input or "피드백" in user_input or ".txt" in user_input:
        state['input_type'] = "document"
        state['uploaded_document'] = "(지원서 내용)"
        print(" > 유형: 문서 피드백")
    else:
        state['input_type'] = "question"
        state['question'] = user_input
        print(" > 유형: 뉴스 Q&A")
    return state

def retrieve_chunks_node(state: GraphState):
    print("--- 2a. 관련 뉴스 기사 검색 ---")
    if retriever is None:
        print(" > 검색기가 초기화되지 않아, 전체 기사를 컨텍스트로 사용합니다.")
        state['relevant_chunks'] = [state['news_article']]
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
    print(f" > 생성된 답변: {answer[:30]}...")
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
        print(" > 검증 결과: 통과 (내용에 근거함)")
    else:
        state['is_grounded'] = False
        print(" > 검증 결과: 실패 (내용에 근거하지 않음)")
    return state


def generate_feedback_node(state: GraphState):
    print("--- 2b. 문서 피드백 생성 (대화 맥락 반영) ---")
    prompt = ChatPromptTemplate.from_template(
        """당신은 IT 기업의 채용 전문가입니다.
        [이전 대화 내용]을 참고하고, 아래 [뉴스 기사]의 핵심 내용을 바탕으로 제출된 [지원서]의 강점과 약점을 분석하고 개선점을 제안해주세요.

        [이전 대화 내용]:
        {chat_history}

        [뉴스 기사]:
        {news_article}

        [지원서]:
        {document}
        """
    )
    feedback_chain = prompt | llm | StrOutputParser()
    feedback = feedback_chain.invoke({
        "chat_history": "\n".join([f"{msg.type}: {msg.content}" for msg in state['chat_history']]),
        "news_article": state['news_article'],
        "document": state['uploaded_document']
    })
    state['feedback'] = feedback
    print(f" > 생성된 피드백: {feedback[:30]}...")
    return state

# --- 4. 그래프 구성 ---
workflow = StateGraph(GraphState)

workflow.add_node("classify_input", classify_input_node)
workflow.add_node("retrieve_chunks", retrieve_chunks_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("grade_answer", grade_answer_node)
workflow.add_node("generate_feedback", generate_feedback_node)

workflow.set_entry_point("classify_input")

workflow.add_conditional_edges(
    "classify_input",
    lambda state: state["input_type"],
    {
        "question": "retrieve_chunks",
        "document": "generate_feedback",
    }
)

workflow.add_edge("retrieve_chunks", "generate_answer")
workflow.add_edge("generate_answer", "grade_answer")
workflow.add_conditional_edges(
    "grade_answer",
    lambda state: "grounded" if state["is_grounded"] else "not_grounded",
    {
        "grounded": END,
        "not_grounded": END, 
    }
)
workflow.add_edge("generate_feedback", END)

app = workflow.compile()

# --- 5. 그래프 실행 (대화 시뮬레이션) ---
if __name__ == "__main__":
    # 그래프 시각화
    try:
        graph_image_path = "langgraph_structure.png"
        with open(graph_image_path, "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print(f"LangGraph 구조가 '{graph_image_path}' 파일로 저장되었습니다.")
    except Exception as e:
        print(f"그래프 시각화 중 오류 발생: {e}")

    if not llm or not news_article:
        print("모델 또는 뉴스 기사를 초기화할 수 없어 실행을 중단합니다.")
    else:
        chat_history = []

        print("\n--- 채팅 시작 (종료하려면 'exit' 또는 '종료' 입력) ---")
        while True:
            user_input = input("\n[사용자]: ")
            if user_input.lower() in ['exit', '종료']:
                print("채팅을 종료합니다.")
                break

            inputs = {
                "user_input": user_input,
                "news_article": news_article,
                "chat_history": chat_history
            }
            
            # LangGraph 실행
            result = app.invoke(inputs)
            
            # 결과 처리 및 출력
            if result.get('input_type') == "question":
                ai_response = result.get('answer', '답변을 생성하지 못했습니다.')
            elif result.get('input_type') == "document":
                ai_response = result.get('feedback', '피드백을 생성하지 못했습니다.')
            else:
                ai_response = "알 수 없는 입력 유형입니다."

            print(f"[AI]: {ai_response}")
            chat_history.extend([HumanMessage(content=user_input), AIMessage(content=ai_response)])