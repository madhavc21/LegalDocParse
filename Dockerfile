# Dockerfile

# Stage 1: Base image with Python
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR off
ENV PIP_DISABLE_PIP_VERSION_CHECK on

# Install system dependencies that might be needed by some Python packages (e.g., lxml)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# IMPORTANT: If you intend to use a GPU with Docker, you might want to install
# a GPU-enabled PyTorch version here. The requirements.txt lists CPU by default.
# For GPU PyTorch (example for CUDA 11.8, adjust as needed):
# RUN pip install --no-cache-dir -r requirements.txt \
#    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU (as per current requirements.txt):
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application stage (copies from base to keep image smaller)
FROM base as application

WORKDIR /app

# Copy the rest of the application code
COPY . .

# Download and install spaCy models
# This assumes your requirements.txt has 'spacy'
RUN python -m spacy download en_core_web_lg

# Install the custom legal NER model from its wheel file
# The URL points to the specific version from OpenNyAI
RUN pip install --no-cache-dir  https://huggingface.co/ali6parmak/en_legal_ner_trf/resolve/main/en_legal_ner_trf-3.2.0-py3-none-any.whl

# Expose the port the app runs on
EXPOSE 8000

# Healthcheck (Optional but good practice)
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#   CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
# Using 0.0.0.0 to be accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]