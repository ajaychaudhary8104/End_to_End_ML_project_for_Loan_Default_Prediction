FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends awscli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser

COPY --chown=appuser:appuser . /app

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Switch to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]