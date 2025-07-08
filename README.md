# llm-svc


# 기능 개발

1. feature/{기능 별칭} 브랜치를 dev로부터 만들기

2. 개발 & 테스트

3. dev에 merge

4. > merge 충돌 주의

# Docker 가이드

===============================================================

## Docker Image 빌드 및 실행

### 1. Docker Hub 에서 llm-service 이미지 pull 받기

```bash
docker pull kwonsoonmin/llm-service:dev0.1
```

### 2. docker-compose 실행

빌드된 이미지를 사용하여 Docker 컨테이너를 실행합니다:

```bash
docker-compose up -d
```

이제 `http://localhost:8000` 에서 LLM Service에 접속할 수 있습니다.

실행 중인 서비스를 중지하려면 다음 명령어를 사용합니다:

```bash

docker-compose down
```

===============================================================
