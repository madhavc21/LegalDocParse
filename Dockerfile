FROM python:3.11 AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
# IMPORTANT: If you intend to use a GPU with Docker, you might want to install
# a GPU-enabled PyTorch version here. The requirements.txt lists CPU by default.
# For GPU PyTorch (example for CUDA 11.8, adjust as needed):
# RUN pip install --no-cache-dir -r requirements.txt \
#    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU (as per current requirements.txt):
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS application

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir  https://huggingface.co/ali6parmak/en_legal_ner_trf/resolve/main/en_legal_ner_trf-3.2.0-py3-none-any.whl

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]