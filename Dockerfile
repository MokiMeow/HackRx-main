# Use official minimal base image
FROM python:3.10-slim

# Install system-level packages needed for PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install torch CPU version manually
COPY requirements.txt .

# Install torch CPU separately to avoid CUDA bloating
RUN pip install --no-cache-dir torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install rest of the deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY ./src ./src

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
