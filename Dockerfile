# ============================================================
#  Dockerfile — Fraud Detection API
# ============================================================

FROM python:3.11-slim

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY train.py run_api.py ./

# Create directories
RUN mkdir -p artifacts logs

# Expose API port
EXPOSE 8000

# Default: run the API (assumes model is already trained)
CMD ["python", "run_api.py"]
