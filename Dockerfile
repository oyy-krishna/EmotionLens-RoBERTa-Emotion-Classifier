FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download demoji codes at build time
RUN python -c "import demoji; demoji.download_codes()"

# Copy the rest of the application code
COPY . .

# HF Spaces runs as port 7860
EXPOSE 7860

# Run the FastAPI app (assumes your main file is main.py with a FastAPI app instance named 'app')
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
