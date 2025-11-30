FROM python:3.10-slim

# Install system dependencies including exiftool
RUN apt-get update && apt-get install -y \
    libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    requests \
    pandas

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
