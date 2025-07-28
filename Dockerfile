# Optimized Python 3.10 Docker image for fast builds
FROM python:3.10-slim

# Set environment variables for faster pip installs
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip

# Install Python packages in optimal order (most stable first)
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install --no-cache-dir scikit-learn pandas networkx PyMuPDF
RUN pip install --no-cache-dir transformers
RUN pip install --no-cache-dir sentence-transformers
RUN pip install --no-cache-dir spacy

# Download models (cached layer)
RUN python -m spacy download en_core_web_sm
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy application code (last to leverage cache)
COPY . .

# Create volume mount point
VOLUME ["/app"]

CMD ["python", "main.py"]
