# Build the existing React UI without changing its design.
FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Keep the free-tier runtime lean: core ATS analysis works without local Ollama,
# ChromaDB, or PyTorch. RAG is explicitly disabled on this constrained instance.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAG_ENABLED=false \
    SESSION_COOKIE_SECURE=true
WORKDIR /app
COPY backend/requirements-render.txt /app/backend/requirements-render.txt
RUN pip install --no-cache-dir -r /app/backend/requirements-render.txt
COPY backend/ /app/backend/
COPY --from=frontend-build /frontend/build/ /app/frontend/build/
WORKDIR /app/backend
EXPOSE 10000
CMD ["sh", "-c", "exec gunicorn app:app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 120 --access-logfile -"]
