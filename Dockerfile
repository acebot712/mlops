FROM python:3.8-slim

# Install ONNX runtime and FastAPI dependencies
RUN pip install onnxruntime fastapi uvicorn Pillow python-multipart torchvision

# Create a directory for the app and set it as the working directory
WORKDIR /app

# Copy model and script into the container
COPY model.onnx /app/model.onnx
COPY serve_model.py /app/serve_model.py

# Expose the port the application will run on
EXPOSE 8000

# Start the FastAPI application using Uvicorn
CMD ["uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
