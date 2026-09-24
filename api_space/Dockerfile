# The RegRAG web API, as a Hugging Face Docker Space.
# Assembled by `python space/sync.py --target api`; see DEPLOY.md.

FROM python:3.13-slim

# Hugging Face runs Space containers as uid 1000, so the app, the pip installs
# and the model cache all belong to that user rather than root.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only torch first, from PyTorch's own index. The default wheel bundles CUDA
# (~2.5GB) for a GPU this Space doesn't have; the CPU build is a fraction of it.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# Download both models while building, not on the first request: a cold start
# then loads them from disk instead of fetching ~150MB from the Hub.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user . .

# One worker, on purpose: see DECISIONS.md #11.
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
