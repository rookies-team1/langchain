import chromadb

collection_name = "news_vector_db"
news_id_to_check = "1"   # 보고 싶은 뉴스 ID (str)

# 1️⃣ ChromaDB 클라이언트 생성
client = chromadb.HttpClient(host="localhost", port=8001)

# 2️⃣ 컬렉션 가져오기
collection = client.get_collection(name=collection_name)

# 3️⃣ news_id로 필터링하여 가져오기
results = collection.get(
    where={"news_id": news_id_to_check},
    include=["documents", "metadatas"]
)

# 4️⃣ 결과 확인
for idx, doc in enumerate(results["documents"]):
    print(f"\n=== 청크 {idx + 1} ===")
    print(f"Metadata: {results['metadatas'][idx]}")
    print(f"Content: {doc[:500]}...")   # 너무 길면 500자까지만 출력
