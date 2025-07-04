# python-app/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Poetry 의존성 설치
COPY pyproject.toml poetry.lock* /app/
RUN pip install poetry
RUN poetry config virtualenvs.create false && poetry install --no-root --only main

# 애플리케이션 코드 복사
COPY ./llm-service /app/llm-service

# EXPOSE는 docker-compose에서 관리하므로 필수는 아님
EXPOSE 8000

# uvicorn을 실행
CMD ["poetry", "run", "uvicorn", "llm-service.main:app", "--host", "0.0.0.0", "--port", "8000"]