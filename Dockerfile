FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

RUN pip install poetry
RUN poetry config virtualenvs.create false && poetry install --no-root

COPY ./llm-service /app/llm-service

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "llm-service.main:app", "--host", "0.0.0.0", "--port", "8000"]