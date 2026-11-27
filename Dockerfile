# Raga Search: API + web app in one container.
#   docker build -t ragasearch .
#   docker run -p 7860:7860 ragasearch      -> http://localhost:7860
FROM python:3.13-slim

# ffmpeg decodes formats libsndfile can't (m4a voice memos, webm).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (uid 1000 is also what Hugging Face Spaces expects).
RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/ragasearch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app ragaid ragaid
COPY --chown=app:app api api
COPY --chown=app:app app app
COPY --chown=app:app models/raga_saraga_hindustani_v1.joblib models/

USER app
ENV PYTHONUNBUFFERED=1 PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"
# Hosts like Render inject $PORT; Hugging Face Spaces uses 7860.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
