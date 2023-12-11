from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
from torchvision import transforms

# List of class labels for CIFAR-10
class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

app = FastAPI()

# Load ONNX model
ort_session = ort.InferenceSession("model.onnx")

# Define the transformation used during the model training
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resizing to 32x32
    transforms.ToTensor(),  # Converting to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Convert image to the correct format
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Apply the same transformations as during training
    img_tensor = transform(image)

    # Add batch dimension if needed
    img_tensor = img_tensor.unsqueeze(0)

    # Run inference
    inputs = {ort_session.get_inputs()[0].name: img_tensor.numpy()}
    outputs = ort_session.run(None, inputs)

    # Process outputs
    probabilities = outputs[0]
    predicted_label_index = np.argmax(probabilities)
    
    # Get the class label name based on the index
    predicted_label = class_labels[predicted_label_index]

    # Return both the predicted label and the probabilities
    return {"predicted_label": predicted_label, "probabilities": probabilities.tolist()}
