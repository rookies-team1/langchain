import chromadb

# 1. 로컬 모드 클라이언트 생성
client = chromadb.HttpClient(host="localhost", port=8001)

# 2. 기존 collection 가져오기
collection = client.get_or_create_collection(name="user_resume_db")

# 3. 저장된 ID로 데이터 조회
result = collection.get(ids=["2_2"])

# 4. 내용 확인
print("✅ 저장된 ID:", result['ids'])
print("✅ 저장된 문서 요약:")
print(result['documents'][0])
print("✅ 저장된 메타데이터:")
print(result['metadatas'][0])