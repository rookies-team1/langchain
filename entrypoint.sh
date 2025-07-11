#!/bin/bash
set -e

# Ollama 서버를 백그라운드에서 실행
echo "Starting Ollama server..."
ollama serve &

# 서버가 준비될 때까지 대기 (Health Check)
echo "Waiting for Ollama server to be ready..."
while ! curl -s -f http://ollama:11434/ > /dev/null; do
    echo "Ollama server not yet available, waiting..."
    sleep 1
done
echo "Ollama server is ready."

# 메인 Python 애플리케이션 실행 (FastAPI/Uvicorn 예시)
# 이 부분이 실제 서비스의 핵심입니다.
echo "Starting the Python LLM service on port 8000..."
exec uvicorn llm-service.main:app --host 0.0.0.0 --port 8000