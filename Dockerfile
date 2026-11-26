FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir .
COPY config ./config
COPY prompts ./prompts
CMD ["uvicorn", "felops.api.main:app", "--host", "0.0.0.0", "--port", "8000"]