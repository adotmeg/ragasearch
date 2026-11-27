# Hugging Face Space: builds Raga Search from the GitHub repository.
FROM python:3.13-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
USER app
WORKDIR /home/app

# Download the source at a branch/tag/commit (no git needed).
ARG REPO=adotmeg/ragasearch
ARG REF=main
RUN python -c "import io, tarfile, urllib.request; \
url = 'https://github.com/${REPO}/archive/${REF}.tar.gz'; \
tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read())).extractall('src', filter='data')" \
 && mv src/* ragasearch && rmdir src

WORKDIR /home/app/ragasearch
RUN pip install --no-cache-dir --user -r requirements.txt
ENV PATH=/home/app/.local/bin:$PATH PYTHONUNBUFFERED=1 PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
