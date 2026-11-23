FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 EURS_MODELS_DIR=/app/models
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY eurs_core ./eurs_core
COPY app ./app
COPY models ./models
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
