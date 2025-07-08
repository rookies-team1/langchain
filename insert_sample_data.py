
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import chromadb

# --- 설정 (llm-service/chat_langgraph_2.py와 동일하게 유지) ---

def get_embeddings():
    """임베딩 모델 인스턴스를 가져옵니다."""
    load_dotenv()
    try:
        return OllamaEmbeddings(
            model="bge-m3:567m",
            base_url="http://localhost:11434"
        )
    except Exception as e:
        print(f"Ollama 임베딩 모델 로드 실패: {e}")
        print("Ollama 컨테이너가 실행 중인지, OLLAMA_BASE_URL 환경 변수가 올바르게 설정되었는지 확인하세요.")
        raise

def get_chroma_client():
    """ChromaDB 클라이언트 인스턴스를 가져옵니다."""
    load_dotenv()
    CHROMA_HOST = "localhost"
    CHROMA_PORT = int(os.getenv("VECTOR_DB_PORT", "8001"))
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat() # 연결 테스트
        return client
    except Exception as e:
        print(f"ChromaDB 연결 실패: {e}")
        print(f"{CHROMA_HOST}:{CHROMA_PORT} 주소로 ChromaDB에 연결할 수 없습니다.")
        print("ChromaDB 컨테이너가 실행 중인지, VECTOR_DB_HOST/PORT 환경 변수가 올바르게 설정되었는지 확인하세요.")
        raise

# --- 예시 데이터 ---

sample_news = [
    {
        "news_id": "101",
        "title": "SK쉴더스, 제로 트러스트 기반 신규 보안 모델 공개",
        "content": """
        SK쉴더스가 클라우드, 인공지능(AI) 시대에 맞춰 새로운 보안 패러다임인 '제로 트러스트(Zero Trust)' 모델을 발표했습니다. 
        이 모델은 '절대 신뢰하지 말고, 항상 검증하라(Never Trust, Always Verify)'는 원칙을 기반으로 합니다.
        기존의 경계 기반 보안 모델은 내부 네트워크에 접속한 사용자를 신뢰하는 방식이었지만, 
        클라우드와 원격 근무 환경이 보편화되면서 이러한 방식은 한계에 부딪혔습니다.
        SK쉴더스의 제로 트러스트 모델은 모든 사용자, 기기, 애플리케이션의 접근 요청을 잠재적인 위협으로 간주하고, 
        매번 철저한 인증과 권한 검사를 수행하여 보안을 강화합니다. 
        이를 통해 내부자 위협이나 계정 탈취 공격에도 효과적으로 대응할 수 있습니다.
        """
    },
    {
        "news_id": "202",
        "title": "네이버, 차세대 초거대 AI '하이퍼클로바X' 공개",
        "content": """
        네이버가 차세대 초거대 인공지능(AI) 모델인 '하이퍼클로바X'를 공개하며 AI 시장에 새로운 바람을 일으키고 있습니다.
        하이퍼클로바X는 이전 모델보다 더욱 방대한 데이터를 학습했으며, 특히 한국어에 대한 이해와 생성 능력이 뛰어납니다.
        네이버는 이 모델을 검색, 쇼핑, 광고 등 자사 서비스 전반에 적용하여 사용자 경험을 혁신할 계획입니다.
        또한, 개발자들이 하이퍼클로바X를 활용하여 다양한 AI 서비스를 만들 수 있도록 API를 공개하고, 
        기업 고객을 위한 맞춤형 AI 솔루션도 제공할 예정입니다. 
        이를 통해 국내 AI 생태계를 확장하고 글로벌 시장에서의 경쟁력을 확보하겠다는 전략입니다.
        """
    }
]

def main():
    """예시 데이터를 ChromaDB에 저장하는 메인 함수"""
    print("--- ChromaDB에 예시 데이터 저장을 시작합니다. ---")
    
    try:
        # 1. ChromaDB 및 임베딩 모델 클라이언트 가져오기
        embeddings = get_embeddings()
        chroma_client = get_chroma_client()
        collection_name = "news_vector_db"
        
        print(f"✅ 임베딩 모델 및 ChromaDB 클라이언트 연결 성공")

        # 2. 텍스트 분할 및 Document 객체 생성
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        all_docs = []
        
        for news in sample_news:
            chunks = text_splitter.split_text(news["content"])
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "news_id": news["news_id"],
                        "title": news["title"],
                        "chunk_index": i 
                    }
                )
                all_docs.append(doc)
        
        print(f"✅ 총 {len(sample_news)}개의 뉴스를 {len(all_docs)}개의 청크로 분할했습니다.")

        # 3. ChromaDB에 데이터 저장
        print(f"⏳ '{collection_name}' 컬렉션에 데이터 저장 중...")
        
        # 기존 컬렉션이 있다면 삭제하고 새로 생성 (멱등성 보장)
        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"   - 기존 '{collection_name}' 컬렉션을 삭제했습니다.")
        except Exception:
            print(f"   - 기존 '{collection_name}' 컬렉션이 없어 새로 생성합니다.")

        Chroma.from_documents(
            documents=all_docs,
            embedding=embeddings,
            client=chroma_client,
            collection_name=collection_name
        )
        
        print(f"✅ '{collection_name}' 컬렉션에 데이터 저장을 완료했습니다.")
        
        # 4. 저장된 데이터 확인 (선택 사항)
        collection = chroma_client.get_collection(name=collection_name)
        count = collection.count()
        print(f"   - 확인: 현재 컬렉션에 저장된 문서 수는 {count}개 입니다.")
        
        # news_id 101로 필터링하여 개수 확인
        count_101 = len(collection.get(where={"news_id": "101"})['ids'])
        print(f"   - 확인: news_id '101'에 해당하는 문서는 {count_101}개 입니다.")


    except Exception as e:
        print(f"🔥 데이터 저장 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
