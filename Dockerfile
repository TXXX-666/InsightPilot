FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY python/pyproject.toml /app/python/pyproject.toml
COPY python/insightpilot /app/python/insightpilot
COPY frontend /app/frontend
COPY .insightpilot /app/.insightpilot
COPY assets /app/assets

RUN python -m pip install --upgrade pip \
    && python -m pip install /app/python

RUN mkdir -p /app/data /app/reports

EXPOSE 8000 8501

CMD ["uvicorn", "insightpilot.api:app", "--app-dir", "/app/python", "--host", "0.0.0.0", "--port", "8000"]
