FROM python:3.11-slim

WORKDIR /app

# System deps for audio + document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p vectorstore data/documents

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
