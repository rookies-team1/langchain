FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

RUN pip install poetry
RUN poetry config virtualenvs.create false && poetry install --no-root

COPY ./llm-service /app/llm-service

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

CMD ["/app/entrypoint.sh"]