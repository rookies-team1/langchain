import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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
    """)                                     

llm = Ollama(model = "qwen3:1.7b")

# llm = ChatOllama(model="bge-m3:latest")

# chain 연결 (LCEL) prompt + llm + outputparser
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


# chain 호출
# try:
#     result = chain.invoke(news_json)
#     print("요약 결과:", result)
# except Exception as e:
#     print(f"오류 발생: {e}")

def summarize_news(news_json: dict) -> str:
    return chain.invoke(news_json)