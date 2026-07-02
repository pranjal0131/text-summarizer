# Multi-stage build keeps the final image small.
FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY summarizer/ summarizer/
COPY api/ api/
RUN pip install --no-cache-dir --prefix=/install ".[api]"

FROM python:3.12-slim

# Run as a non-root user.
RUN useradd --create-home appuser
COPY --from=builder /install /usr/local
USER appuser
WORKDIR /home/appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
