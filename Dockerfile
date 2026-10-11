FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    DATA_DIR=/data

WORKDIR /srv

# CPU-only torch keeps the image far smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the first upload needs no download.
# The model name comes from config.py, the single source of truth.
COPY app/config.py app/config.py
RUN cd app && python -c "from sentence_transformers import SentenceTransformer; from config import EMBED_MODEL; SentenceTransformer(EMBED_MODEL)"

COPY app app
COPY .streamlit .streamlit

RUN useradd --create-home appuser \
    && mkdir -p /data \
    && chown -R appuser /data
USER appuser

VOLUME /data
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/main.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
