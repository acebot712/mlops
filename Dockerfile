# Use a specific python slim image version for reproducibility
FROM python:3.8-slim as base

# Stage 1: Build environment
FROM base as builder

# Create a directory for the app
WORKDIR /app

# Copy model and script into the builder image
COPY model.onnx serve_model.py ./

# Stage 2: Runtime environment
FROM base

# Install runtime dependencies
RUN pip install --no-cache-dir onnxruntime fastapi uvicorn Pillow python-multipart torchvision \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files from the builder stage
COPY --from=builder /app /app

# Set the working directory
WORKDIR /app

# Expose the port the application will run on
EXPOSE 8000

# Start the FastAPI application using Uvicorn
CMD ["uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
