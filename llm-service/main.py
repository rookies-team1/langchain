from fastapi import FastAPI
import os

app = FastAPI(title="AI Agent API")

# 환경 변수에서 API 키 로드
google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
 
@app.get("/")
def read_root():
    if google_api_key:
        return {"message": "LLM Service is running.", "api_key_status": "loaded", "key" : google_api_key[:2]}
    else:
        return {"message": "LLM Service is running.", "api_key_status": "not loaded"}