# Use PyTorch base image with support for CPU/GPU
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies (needed for OpenCV and git)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Create folders for uploads and results
RUN mkdir -p uploads results

# Expose port
EXPOSE 5000

# Start Flask app
CMD ["python", "app.py"]
