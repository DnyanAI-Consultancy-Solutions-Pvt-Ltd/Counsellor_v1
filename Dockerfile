FROM python:3.11-slim

# -----------------------------
# Environment Variables
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Streamlit
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# API
ENV API_URL=http://127.0.0.1:8000

# Chroma
ENV CHROMA_DB_PATH=/home/user/app/storage/chroma_db

# -----------------------------
# Create non-root user
# -----------------------------
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# -----------------------------
# Linux packages
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    poppler-utils \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install Python packages
# -----------------------------
COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

# -----------------------------
# Copy project
# -----------------------------
COPY --chown=user:user . .

# -----------------------------
# Create runtime directories
# -----------------------------
RUN mkdir -p \
    storage/chroma_db \
    storage/uploads \
    storage/output \
    sessions \
    && chmod +x run.sh \
    && chown -R user:user /home/user/app

USER user

EXPOSE 7860

CMD ["./run.sh"]