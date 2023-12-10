from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load ONNX model
ort_session = ort.InferenceSession("model.onnx")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Convert image to the correct format
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((32, 32)).convert('RGB')
    img_array = np.array(image).astype('float32') / 255
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = img_array.reshape(1, 3, 32, 32)

    # Run inference
    inputs = {ort_session.get_inputs()[0].name: img_array}
    outputs = ort_session.run(None, inputs)

    # Process outputs (this is simplified; you would add your own processing logic)
    prediction = np.argmax(outputs[0])

    return {"prediction": int(prediction)}
