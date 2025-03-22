from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import onnxruntime as ort
import numpy as np
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
import joblib  # To load scaler
import base64

app = FastAPI()

# Load ONNX model
ort_session = ort.InferenceSession("bmi_model.onnx")

# Load scaler
scaler = joblib.load("RobustScaler.pkl")

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

@app.get("/", response_class=HTMLResponse)
async def show_form():
    return """
    <html>
        <head>
            <title>BMI Predictor</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
            <style>
                body { text-align: center; margin: 50px; background-color: #f8f9fa; }
                .container { max-width: 500px; background: white; padding: 30px; border-radius: 10px; box-shadow: 0px 0px 10px gray; }
                h2 { color: #007bff; }
                .btn-primary { background-color: #007bff; border: none; }
                .btn-primary:hover { background-color: #0056b3; }
                img { max-width: 100%; margin-top: 20px; border-radius: 10px; }
                .disclaimer { font-size: 12px; color: #666; margin-top: 10px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Upload an Image to Predict BMI</h2>
                <form action="/predict/" method="post" enctype="multipart/form-data" onsubmit="previewImage()">
                    <input class="form-control" type="file" name="file" id="file" required onchange="previewImage()">
                    <button class="btn btn-primary mt-3" type="submit">Predict</button>
                </form>
                <img id="preview" src="" style="display: none;">
            </div>

            <script>
                function previewImage() {
                    var file = document.getElementById("file").files[0];
                    var reader = new FileReader();
                    reader.onloadend = function () {
                        document.getElementById("preview").src = reader.result;
                        document.getElementById("preview").style.display = "block";
                    }
                    if (file) { reader.readAsDataURL(file); }
                }
            </script>
        </body>
    </html>
    """

@app.post("/predict/", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)
    
    # Use the actual input name from the ONNX model
    input_name = ort_session.get_inputs()[0].name  

    # Run ONNX model inference
    _, outputs = ort_session.run(None, {input_name: input_tensor.numpy()})

    predicted_bmi = outputs[0][0]

    # Apply inverse transform to get actual BMI
    predicted_bmi = scaler.inverse_transform(np.array(predicted_bmi).reshape(-1, 1)).flatten()[0]
    
    # Round to 2 decimal places
    predicted_bmi = f"{predicted_bmi:.2f}"

    # Convert image to base64 for displaying
    encoded_img = base64.b64encode(image_bytes).decode('utf-8')
    img_src = f"data:image/jpeg;base64,{encoded_img}"

    return f"""
    <html>
        <head>
            <title>BMI Predictor</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
            <style>
                body {{ text-align: center; margin: 50px; background-color: #f8f9fa; }}
                .container {{ max-width: 500px; background: white; padding: 30px; border-radius: 10px; box-shadow: 0px 0px 10px gray; }}
                h2 {{ color: #007bff; }}
                img {{ max-width: 100%; margin-top: 20px; border-radius: 10px; }}
                .disclaimer {{ font-size: 12px; color: #666; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Predicted BMI: <b>{predicted_bmi}</b></h2>
                <p class="disclaimer">⚠️ Disclaimer: This BMI prediction is based on a model trained with limited data. The result may not be fully accurate.</p>
                <img src="{img_src}">
                <a href="/" class="btn btn-primary mt-3">Upload Another Image</a>
            </div>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    import nest_asyncio

    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)

