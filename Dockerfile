FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY api/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY api /app/api
COPY artifacts /app/artifacts
COPY rag /app/rag

EXPOSE 5000

CMD ["python", "api/app.py"]
