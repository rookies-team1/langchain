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
    이력서를 준비 중인 취업 준비생이 해당 기업을 분석하는 데 도움이 되도록, 
    핵심 내용을 3~4문장으로 요약해 주세요.

    요약은 다음 기준에 맞게 작성해 주세요:
    - 과장 없이, 사실 기반으로
    - 투자자가 아닌 취준생의 시선에서 기업을 이해하는 데 도움되도록

    [제목]
    {title}

    [내용]
    {content}
    """)                                     

# Groq API를 사용하는 ChatOpenAI 인스턴스 생성
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.groq.com/openai/v1",  # Groq API 엔드포인트
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7
)

# llm = ChatOllama(model="bge-m3:latest")


news_json = {
    "title": "기지개 켠 삼성전자 업고 코스피 상승 출발",
    "content": """
    코스피시장 대장주인 삼성전자 주가 강세를 보이면서 코스피지수가 4일 상승 출발했다. 미국에서 반도체 세액 공제 등을 담은 ‘하나의 크고 아름다운 법(One Big Beautiful Bill Act·OBBBA)’이 의회 문턱을 넘은 영향으로 보인다.
    코스피지수는 4일 오전 9시 9분 3120.47을 나타냈다. 전날보다 4.2포인트(0.13%) 올랐다.
    코스피시장에서 삼성전자와 SK하이닉스가 강세를 보이고 있다. 도널드 트럼프 미국 대통령이 공을 들여 온 OBBBA가 서명을 앞두고 있는 점이 긍정적으로 작용한 것으로 풀이된다. 이 법에는 미국 내 반도체 시설·장비 투자에 대한 세액공제율을 기존 25%에서 35%로 확대하는 내용이 담겼다.
    반대로 이차전지 업종은 불똥이 튀었다. OBBBA에 인플레이션감축법(IRA)에 근거한 전기차 관련 세액공제 폐지 시점을 2032년 말에서 올해 9월 말로 앞당기는 내용이 있기 때문이다.
    LG에너지솔루션은 약세를 보이고 있다. 이 밖에 코스피시장 시가총액 상위 종목 중 현대차, 두산에너빌리티, 기아는 주가가 오름세다. 삼성바이오로직스와 KB금융 등은 전날보다 낮은 가격에 주식이 거래 중이다.
    이차전지 업종 비중이 큰 코스닥지수는 약세를 보이고 있다. 같은 시각 전날보다 1.79포인트(0.23%) 하락한 791.54를 기록했다.
    코스닥시장 시가총액 상위 종목 가운데 에코프로비엠, 에코프로 등이 2% 안팎의 내림세를 나타냈다. 알테오젠과 펩트론, 리가켐바이오 등도 주가가 하락 중이다. 반대로 HLB, 레인보우로보틱스, 파마리서치는 오름세다.
    밤사이 미국 뉴욕증시는 6월 비농업 부문 고용이 시장 예상치를 웃돌며 경기 둔화 우려를 덜어낸 덕분에 오름세를 보였다. 다우존스30산업평균지수는 0.77% 상승했다. 스탠더드앤드푸어스(S&P)500지수와 나스닥지수는 각각 0.83%, 1.02% 올랐다.
    다만 트럼프 대통령이 오는 4일(현지 시각)부터 각국에 관세율을 명시한 서한을 발송하겠다고 밝히면서 불확실성이 여전히 남아있다. 상호 관세 유예 기간은 오는 8일까지다.
    """
}


# chain 연결 (LCEL) prompt + llm + outputparser
output_parser = StrOutputParser()

chain = prompt | llm | output_parser



# chain 호출
try:
    result = chain.invoke(news_json)
    print("요약 결과:", result)
except Exception as e:
    print(f"오류 발생: {e}")