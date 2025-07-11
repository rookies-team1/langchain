# 첫 번째 스테이지: 빌드용
FROM python:3.12-slim AS builder

# 작업 디렉토리를 /app으로 설정
WORKDIR /app

# 패키지 빌드에 필요한 시스템 의존성 설치 및 apt 캐시 삭제
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Poetry (의존성 관리 도구) 설치
RUN pip install poetry

# Poetry가 프로젝트 폴더 내에 가상 환경(.venv)을 생성하도록 설정
RUN poetry config virtualenvs.in-project true

# 의존성 정의 파일(pyproject.toml, poetry.lock)을 컨테이너로 복사
COPY pyproject.toml poetry.lock* /app/

# pyproject.toml에 명시된 메인 의존성만 설치
RUN poetry install --no-root --only main

# 두 번째 스테이지: 최종 이미지 - 애플리케이션 실행 환경
FROM python:3.12-slim

# 작업 디렉토리를 /app으로 설정
WORKDIR /app

# 런타임에 필요한 시스템 의존성 설치 및 apt 캐시 삭제
RUN apt-get update && apt-get install -y libopenblas-dev libomp-dev curl && rm -rf /var/lib/apt/lists/*

# 빌더 스테이지에서 생성된 가상 환경을 최종 이미지로 복사
COPY --from=builder /app/.venv ./.venv

# PATH 환경 변수에 가상 환경의 bin 디렉토리를 추가하여, poetry run 없이도 실행 가능하게 함
ENV PATH="/app/.venv/bin:$PATH"

# 애플리케이션 코드와 엔트리포인트 스크립트를 컨테이너로 복사
COPY ./llm-service /app/llm-service
COPY entrypoint.sh /app/entrypoint.sh

# 엔트리포인트 스크립트에 실행 권한 부여
RUN chmod +x /app/entrypoint.sh

# 8000번 포트를 외부에 노출
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]

# uvicorn을 사용하여 애플리케이션 실행
# CMD ["uvicorn", "llm-service.main:app", "--host", "0.0.0.0", "--port", "8000"]
