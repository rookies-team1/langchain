
import os
import sys
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from chromadb import chromadb
from langchain_ollama import OllamaEmbeddings

# llm-service 디렉토리를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 환경 변수 로드
load_dotenv()

# ChromaDB 클라이언트 및 임베딩 모델 초기화
CHROMA_HOST = os.getenv("VECTOR_DB_HOST", "localhost")
CHROMA_PORT = int(os.getenv("VECTOR_DB_PORT", "8001"))
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

embeddings = OllamaEmbeddings(
    model="bge-m3:latest", 
    base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
)
print(embeddings)

def save_to_chroma(news_id: int, content: str, collection_name: str = "news_vector_db"):
    """
    뉴스 데이터를 ChromaDB에 저장합니다.
    """
    try:
        print(f"--- news_id='{news_id}'에 대한 데이터 저장 시작 ---")

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = [Document(page_content=chunk, metadata={"news_id": str(news_id)}) 
                for chunk in text_splitter.split_text(content)]

        # ChromaDB에 저장
        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            client=chroma_client,
            collection_name=collection_name
        )
        print(f"--- news_id='{news_id}'에 대한 데이터 저장 완료 ---")

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    # 예시 데이터
    news_id = 101
    content = """
    SK쉴더스는 26일 서울 여의도 FKI타워에서 ‘SK쉴더스 미디어 세미나’를 열고 생성형 인공지능(AI) 시대의 새로운 보안위협과 대응 전략을 발표했다.

    이날 SK쉴더스는 ‘제로 트러스트’ 모델을 제시했다. 제로 트러스트는 ‘아무도 신뢰하지 않고, 항상 검증한다’는 개념에서 출발한 보안 모델이다. 기존 경계형 보안 모델은 내부자에 대한 신뢰를 기반으로 해 내부에서 발생하는 위협을 탐지하기 어렵다는 한계가 있었다. 반면 제로 트러스트 모델은 모든 사용자·기기·애플리케이션(앱)에 대해 항상 신원을 확인하고 최소한의 권한만 부여해 내부 위협을 차단한다.

    SK쉴더스는 제로 트러스트 모델을 구현하기 위해 자사의 위협 인텔리전스 플랫폼 ‘시큐디움’과 AI 기반 보안 플랫폼 ‘에이닷’을 연동했다. 시큐디움은 국내외 다양한 출처의 위협 정보를 수집·분석해 위협 예측 정보를 제공한다. 에이닷은 AI를 활용해 사용자·기기·앱의 행위를 분석하고 위협을 탐지한다.

    SK쉴더스는 두 플랫폼을 연동해 사용자·기기·앱의 행위와 위협 정보를 종합적으로 분석하고, 이를 기반으로 접근 권한을 동적으로 제어한다. 예를 들어 특정 사용자가 평소와 다른 시간에, 다른 장소에서, 다른 기기로 접속을 시도하면 에이닷이 이를 비정상 행위로 탐지하고, 시큐디움이 해당 사용자의 접속 기록과 위협 정보를 분석해 접근을 차단하는 식이다.

    SK쉴더스는 제로 트러스트 모델을 통해 기업의 내부 정보 유출을 막고, 랜섬웨어 등 외부 공격으로부터 시스템을 보호할 수 있을 것으로 기대하고 있다.
    """

    save_to_chroma(news_id, content)
