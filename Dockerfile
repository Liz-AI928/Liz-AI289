# =================================================================
# Liz AI Dockerfile Optimized for Hugging Face Spaces
# =================================================================

FROM python:3.11-slim

# Set working directory
WORKDIR /code

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        build-essential \
        wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements
COPY ./requirements.txt /code/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy application code
COPY ./api /code/api
COPY ./public /code/public

# Expose default HF Spaces port
EXPOSE 7860

# Environment variables (optional defaults)
ENV HF_TOKEN=""
ENV OPENWEATHERMAP_API_KEY=""
ENV SERPER_API_KEY=""

# Gunicorn + Uvicorn worker with 1 worker, 2-minute timeout for large models
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:7860", "api.app:app", "--workers", "1", "--timeout", "120"]
