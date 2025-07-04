import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


prompt = PromptTemplate(
    input_variables=["title", "content"],
    template="""
    당신은 경제 및 산업 분석에 능한 리서처입니다.

    아래는 한 기업에 대한 뉴스 기사입니다.
    이력서를 준비 중인 취업 준비생이 해당 기업을 분석하는 데 도움이 되도록, 핵심 내용을 3~4문장으로 요약해 주세요.

    요약은 다음 기준에 맞게 작성해 주세요:
    - 기업의 최근 이슈, 실적, 전략, 사업 방향 등을 중심으로
    - 과장 없이, 사실 기반으로
    - 투자자가 아닌 취준생의 시선에서 기업을 이해하는 데 도움되도록

    [제목]
    {title}

    [내용]
    {content}
    """)                                     

# Groq API를 사용하는 ChatOpenAI 인스턴스 생성
# llm = ChatOpenAI(
#     api_key=OPENAI_API_KEY,
#     base_url="https://api.groq.com/openai/v1",  # Groq API 엔드포인트
#     model="meta-llama/llama-4-scout-17b-16e-instruct",
#     temperature=0.7
# )

llm = ChatOllama(model="bge-m3:latest")


news_json = {
    "title": "삼성전자, 2분기 영업이익 8조 돌파",
    "content": """삼성전자가 2025년 2분기 실적을 발표하며 영업이익 8조 원을 기록했다.
    반도체 부문의 회복과 갤럭시 스마트폰 시리즈의 글로벌 판매 호조가 주된 요인으로 분석된다.
    전년 동기 대비 120% 증가한 수치로, 시장 기대치를 상회한 실적이다.
    삼성전자는 하반기에도 메모리 수요 회복세가 지속될 것으로 내다봤다."""
}


# chain 연결 (LCEL) prompt + llm + outputparser
output_parser = StrOutputParser()

chain = prompt | llm | output_parser



# chain 호출
try:
    result = chain.invoke(news_json)
except Exception as e:
    print(f"오류 발생: {e}")