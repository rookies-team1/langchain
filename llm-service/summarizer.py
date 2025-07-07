import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import re

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

llm = OllamaLLM(model = "qwen3:1.7b")

# chain 연결 (LCEL) prompt + llm + outputparser
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


def clean_llm_output(text: str) -> str:
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

# chain 호출
# try:
#     result = chain.invoke(news_json)
#     print("요약 결과:", result)
# except Exception as e:
#     print(f"오류 발생: {e}")

def summarize_news(news_json: dict) -> str:
    raw_output = chain.invoke(news_json)
    return clean_llm_output(raw_output)