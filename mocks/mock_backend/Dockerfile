FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "fastapi==0.156.1" \
    "uvicorn==0.54.0" \
    "pydantic==2.14.1"

COPY . .

EXPOSE 9001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9001"]
