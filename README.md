# llm-svc (dev branch)

### 여기는 dev 브랜치입니다.

# 기능 개발

1. feature/{기능 별칭} 브랜치를 dev로부터 만들기

2. 개발 & 테스트

3. dev에 merge

4. > merge 충돌 주의

# Docker 가이드

===============================================================

## Docker Image 빌드 및 실행

Docker를 사용하여 애플리케이션을 빌드하고 실행할 수 있습니다.

### 1. Docker Image 빌드

프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 Docker 이미지를 빌드합니다:

```bash
docker build -t llm-service .
```

### 2. Docker Container 실행

빌드된 이미지를 사용하여 Docker 컨테이너를 실행합니다:

```bash
docker run -p 8000:8000 llm-service
```

이제 `http://localhost:8000` 에서 LLM Service에 접속할 수 있습니다.

===============================================================

## Docker Compose 사용

`docker-compose`를 사용하여 애플리케이션을 더 쉽게 관리할 수 있습니다.

### 1. Docker Compose로 서비스 시작

프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 서비스를 시작합니다:

```bash
docker-compose up --build
```

`--build` 옵션은 이미지가 없는 경우 자동으로 빌드하고, 변경 사항이 있는 경우 다시 빌드합니다.

### 2. 백그라운드에서 서비스 실행

서비스를 백그라운드에서 실행하려면 `-d` 옵션을 추가합니다:

```bash
docker-compose up -d --build
```

### 3. 서비스 중지

실행 중인 서비스를 중지하려면 다음 명령어를 사용합니다:

```bash

docker-compose down
```

===============================================================
