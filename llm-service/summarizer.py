import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from .chat_langgraph import get_chroma_client, get_embeddings
from langchain_chroma import Chroma
import json
from langsmith import Client
from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio

# ========== LLM 및 처리 체인 초기화 ==========
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt = PromptTemplate(
    input_variables=["title", "content"],
    template="""
    당신은 경제 및 산업 분석에 능한 리서처입니다.

    아래는 한 기업에 대한 뉴스 기사입니다.
    이력서를 준비 중인 취업 준비생이 해당 기업을 분석하는 데 도움이 되도록, 
    핵심 내용을 3~4문장으로 요약해 주세요.

    요약은 다음 기준에 맞게 작성해 주세요:
    - 과장 없이, 사실 기반으로
    - 투자자가 아닌 취준생의 시선에서 기업을 이해하는 데 도움되도록
    - 답변만 출력

    [제목]
    {title}

    [내용]
    {content}
    """    
)

output_parser = StrOutputParser()

# 처리 체인을 미리 구성
summarization_chain = prompt | llm | output_parser

# ========== 유틸 함수 ==========
def clean_llm_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()
    text = re.sub(r"```(?:markdown)?\s*(.*?)```", r"\1", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

@traceable(run_type="chain", name="Simple_Chain")
async def summarize_news(news_json: dict) -> str:
    news_id = str(news_json["id"])
    llm = news_json["llm"]
    embeddings = news_json["embeddings"]

    # 1️⃣ ChromaDB에서 news_id에 해당하는 원문 청크 가져오기
    try:
        chroma_client = get_chroma_client()
        vectorstore = Chroma(
            client=chroma_client,
            collection_name="news_vector_db",
            embedding_function=embeddings,
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={
                "filter": {"news_id": news_id},
                "k": 20  # 필요한 만큼 조정 가능
            }
        )

        # dummy question (LLM 요약용이므로 질문 의미 없음)
        dummy_question = "이 뉴스의 전체 내용을 주세요."
        documents = await retriever.ainvoke(dummy_question)

        if not documents:
            raise ValueError(f"news_id '{news_id}' 에 해당하는 뉴스 원문을 ChromaDB에서 찾을 수 없습니다.")

        # 가져온 청크들을 하나의 문자열로 결합
        content_chunks = []
        for doc in documents:
            raw_content = doc.page_content.strip()
            if not raw_content:
                continue
            # JSON 형태로 저장된 경우 풀어주기
            if raw_content.startswith('{') and raw_content.endswith('}'):
                try:
                    parsed = json.loads(raw_content)
                    extracted_text = parsed.get("data", {}).get("contents", raw_content)
                except Exception:
                    extracted_text = raw_content
            else:
                extracted_text = raw_content

            content_chunks.append(extracted_text)

        combined_content = "\n\n".join(content_chunks)

    except Exception as e:
        print(f"🔥 ChromaDB 검색 및 결합 중 오류: {e}")
        combined_content = "(원문을 불러오지 못했습니다.)"

    # 2️⃣ LLM 요약 호출
    inputs = {
        "title": news_json.get("title", ""),
        "content": combined_content
    }

    raw_output = await summarization_chain.ainvoke(inputs)
    return clean_llm_output(raw_output)

# ========== 테스트용 실행 블록 ==========
if __name__ == "__main__":
    load_dotenv()

    # LangSmith 설정
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "llm-service-already")
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
    client = Client(api_key=LANGSMITH_API_KEY)

    # 테스트용 뉴스 데이터
    test_news = {
        "id": 101,
        "title": "SK쉴더스, 제로 트러스트 모델로 클라우드 보안 강화",
    }

    # 비동기 함수 실행
    async def main():
        from .chat_langgraph import get_llm, get_embeddings
        llm = get_llm()
        embeddings = get_embeddings()
        test_news["llm"] = llm
        test_news["embeddings"] = embeddings
        summary = await summarize_news(test_news)
        print("="*50)
        print("[뉴스 요약 결과]")
        print(summary)
        print("="*50)

    asyncio.run(main())