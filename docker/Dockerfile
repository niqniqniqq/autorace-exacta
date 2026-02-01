FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app/__init__.py app/__init__.py

RUN pip install --no-cache-dir ".[dev]"

COPY . .
RUN pip install --no-cache-dir -e .

RUN mkdir -p /app/data /app/models

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
