# LLM 서비스 프로젝트

## 시작하기

### 필수 요구사항

프로젝트를 실행하기 전에 다음 소프트웨어가 설치 필요

- [Python 3.12+](https://www.python.org/)
- [Poetry](https://python-poetry.org/docs/#installation)

### ⚙️ 설치

1. **Git 리포지토리 클론**

   ```sh
   git clone <your-repository-url>
   cd <your-project-directory>
   ```

2. **Poetry를 이용한 의존성 설치**
   프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 필요한 패키지를 설치합니다. `pyproject.toml` 파일에 정의된 모든 의존성이 가상 환경에 설치됩니다.
   > 경로 확인 필수 (해당 경로에 'pyproject.toml' 파일이 존재해야합니다.)
   ```sh
      poetry install
   ```

## ▶️ 실행

의존성 설치가 완료되면 다음 명령어를 사용하여 개발 서버를 시작할 수 있습니다.

```sh
poetry run uvicorn llm-service.main:app --reload
```

또는
'''sh
docker-compose -d up
'''
도커 이미지로 올려서 실행해도 상관 없습니다.

서버가 성공적으로 시작되면 웹 브라우저에서 `http://127.0.0.1:8000/docs` 로 접속하여 API 문서를 확인하고 테스트할 수 있습니다.

## ✅ 테스트

프로젝트에 포함된 테스트를 실행하려면 다음 명령어를 사용하세요.

```sh
poetry test
```

## 프로젝트 구조

```
.
├── llm-service/      # 메인 애플리케이션 소스 코드
│   ├── __init__.py
│   └── main.py
├── .env              # 환경 변수 설정 파일
├── .gitignore        # Git 추적 제외 파일 목록
├── docker-compose.yml# Docker Compose 설정
├── Dockerfile        # Docker 이미지 빌드 설정
├── poetry.lock       # 의존성 버전 잠금 파일
├── pyproject.toml    # 프로젝트 설정 및 의존성 관리
└── README.md         # 프로젝트 안내 문서
```
